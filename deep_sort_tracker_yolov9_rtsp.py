#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Use deep sort tracker in yoloV9 on either NVIDIA csi or usb camera input or a video file or single picture file 
#
# ref:- deep sort https://github.com/sujanshresstha/YOLOv9_DeepSORT/blob/main/object_tracking.py
#
# requires NVIDIA library as below
# git clone https://github.com/NVIDIA-AI-IOT/jetcam
# cd jetcam
# sudo python3 setup.py install
#
# added rtsp access using this library https://github.com/olkham/IPyCam/blob/main/examples
#
import sys
import cv2
from .camera import Camera                                           # from NVIDIA jetcam library on github
import atexit
import numpy as np
import threading
import traitlets
import os
import signal
import torch
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from ipycam import IPCamera, CameraConfig

RUN_LOOP=2
def sig_hangup_handler(sig, frame):
    sys.stderr.write('sig_hangup_handler({})\n'.format(sig))
    try:
        sys.stderr.write('restarting...\n')
        os.execve('/usr/bin/python3', ['/usr/bin/python3', 'deep_sort_tracker_yolov9.py'], {})
    except OSError as e:
        sys.stderr.write("execve():{}\n".format(e))
        os._exit(1)

def handler(signum, frame):
    global RUN_LOOP
    print(f'handler active im stopping.... (signum={signum})')
    RUN_LOOP = 0

class CSICamera(Camera):
    capture_device = traitlets.Integer(default_value=0)
    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)
    def __init__(self, *args, **kwargs):
        super(CSICamera, self).__init__(*args, **kwargs)
        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            re, image = self.cap.read()
            if not re:
                raise RuntimeError('Could not read image from camera.')
        except:
            raise RuntimeError('Could not initialize camera.  Please see error trace.')
        atexit.register(self.cap.release)

    def __del__(self):
        self.cap.release()

    def _gst_str(self):
        return 'nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                self.capture_device, self.capture_width, self.capture_height, self.capture_fps, self.width, self.height)
    def _read(self):
        re, image = self.cap.read()
        if re:
            return image
        else:
            raise RuntimeError('Could not read image from camera')

class USBCamera(Camera):

    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)   
    capture_device = traitlets.Integer(default_value=0)
    def __init__(self, *args, **kwargs):
        super(USBCamera, self).__init__(*args, **kwargs)
        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            re , image = self.cap.read()  
            if not re:
                raise RuntimeError('Could not read image from camera.')    
        except:
            raise RuntimeError('Could not initialize camera.  Please see error trace.')
        atexit.register(self.cap.release)

    def __del__(self):
        self.cap.release()

    def get_video_params(self):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return frame_width, frame_height, fps
    def _gst_str(self):
        return 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink'.format(self.capture_device, self.capture_width, self.capture_height, self.capture_fps)
    def _read(self):
        re, image = self.cap.read()
        if re:
            image_resized = cv2.resize(image,(int(self.width),int(self.height)))
            return image_resized
        else:
            raise RuntimeError('Could not read image from camera')

# Define command line flags
flags.DEFINE_string('video', './data/test.mp4', 'Path to input video or webcam index (0)')
flags.DEFINE_string('output', './output/output.mp4', 'path to output video')
flags.DEFINE_float('conf', 0.50, 'confidence threshold')
flags.DEFINE_integer('blur_id', None, 'class ID to apply Gaussian Blur')
flags.DEFINE_integer('class_id', None, 'class ID to track')
flags.DEFINE_string('yolo_model', './weights/yolov9-e.pt', './drone2.v8i.yolov9/')

def main(_argv):
    global RUN_LOOP
    signal.signal(signal.SIGHUP, sig_hangup_handler)                      # try re-start on a hang-up
    signal.signal(signal.SIGUSR1, handler)                                # user graceful kill
    signal.signal(signal.SIGUSR2, handler)
    # Initialize the video capture
    video_input = FLAGS.video
    if FLAGS.video.isdigit():                                             # number means csi=10 or usb port number
        cd = int(video_input)
        if cd == 10:                                                      # csi
            cap = CSICamera(width=224, height=224, capture_width=1080, capture_height=720, capture_fps=30)
            frame_width = 1080
            frame_height = 720
            fps = 30
        else:                                                             # usb
            cap = USBCamera(capture_device=cd)
            frame_width, frame_height, fps = cap.get_video_params()
    else:
        s = video_input.split(".")
        if not s[1].find("mp3") == -1 or not s[1].find("wmv") == -1:          # string is a video file 
            cap = cv2.VideoCapture(video_input)                               # input is a video file string filename and path  
            if not cap.isOpened():
                print('Error: Unable to open video source.')
                return
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        else:
            im = cv2.imread(video_input)                                     # single jpg or png assumed 
            frame_width = int(im.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(im.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(im.get(cv2.CAP_PROP_FPS))
            RUN_LOOP = 1

    config = CameraConfig.load("camera_config.json")                        # load the rtsp config
    # Create virtual camera with default settings
    camera = IPCamera(config=config)                                        # create the rtsp streamer object

    if not camera.start():
        print("Failed to start IPyCam object")
        return

    print(f"\n{'='*60}")
    print(f"  IPyCam is running!")
    print(f"{'='*60}")
    print(f"  Web UI:      http://{camera.config.local_ip}:{camera.config.onvif_port}/")
    print(f"  RTSP Stream: {camera.config.main_stream_rtsp}")
    print(f"  ONVIF:       {camera.config.onvif_url}")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop\n")

    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # select device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLO model
    model = DetectMultiBackend(weights=FLAGS.yolo_model,device=device, fuse=True)
    model = AutoShape(model)

    # Load the COCO class labels
    classes_path = "../configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 

    while RUN_LOOP >= 1:
        if RUN_LOOP == 1:
            frame = im
        else:
            ret, frame = cap.read()
            if not ret:
                break
        # Run model on each frame
        results = model(frame)
        detect = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Filter out weak detections by confidence threshold and class_id
            if FLAGS.class_id is None:
                if confidence < FLAGS.conf:
                    continue
            else:
                if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Apply Gaussian Blur
            if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.imshow('YOLOv9 Object tracking', frame)
        writer.write(frame)
        camera.stream(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if RUN_LOOP == 1:                                                                                # single file used 
            RUN_LOOP -= 1                                                                                # if it was a file then exit the iteration
    # Release video capture and writer
    if not FLAGS.video.isdigit():     
        cap.release()
    else:
        del cap 
    camera.stop()
    cv2.destroyAllWindows()        
    writer.release()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

