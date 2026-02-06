#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Use transformers DETR on either NVIDIA csi or usb camera input or a video file or single picture file 
#
# requires NVIDIA library as below
# git clone https://github.com/NVIDIA-AI-IOT/jetcam
# cd jetcam
# sudo python3 setup.py install
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
# pip install absl-py
# pip3 install absl-py
from absl import app, flags
from absl.flags import FLAGS

from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
from IPython.display import HTML, display

RUN_LOOP=2
def sig_hangup_handler(sig, frame):
    sys.stderr.write('sig_hangup_handler({})\n'.format(sig))
    try:
        sys.stderr.write('restarting...\n')
        os.execve('/usr/bin/python3', ['/usr/bin/python3', 'sys.argv[0]'], {})
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
flags.DEFINE_string('model', 'detr', 'choose detr or panoptic')

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

    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DETR model
    # you can specify the revision tag if you don't want the timm dependency
    if not FLAGS.model.find("panoptic") == -1:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
        pan_model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic").eval()
    else:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    while RUN_LOOP >= 1:
        if RUN_LOOP == 1:
            frame = im
        else:
            ret, frame = cap.read()
            if not ret:
                break
        # Run model on each frame
        if not FLAGS.model.find("panoptic") == -1:
            with torch.no_grad():
                inputs = processor(images=frame, return_tensors="pt")
                outputs = pan_model(**inputs)
                processed = processor.post_process_panoptic_segmentation( outputs, target_sizes=[frame.size[::-1]])[0]
            seg_map = processed["segmentation"]           
            segments_info = processed["segments_info"] 
            num_segments = seg_map.max().item() + 1
            palette = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)
            palette[0] = (0,0,0)  
            frame = palette[seg_map.numpy()]                                                                         # (H,W,3) uint8
            id2label = pan_model.config.id2label  
            rows = ""
            for seg in segments_info:
                seg_id  = seg["id"]                                                                                  # SEQ ID
                cid     = seg["label_id"]                                                                            # COCO  ID
                score   = seg["score"]              
                label   = id2label[cid]
                rgb     = palette[seg_id]
                hexcol  = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                rows += f"<tr><td>{seg_id}</td><td>{label}</td><td>{score:.3f}</td><td style='background:{hexcol};width:40px;'></td><td>{hexcol}</td></tr>"    
            html = f"""
            <table>
                <tr><th>seg_id</th><th>label</th><th>score</th><th>color</th><th>hex</th></tr>
                {rows}
            </table>
            """      
            display(HTML(html))            
        else:   
            inputs = processor(images=frame, return_tensors="pt")
            outputs = model(**inputs)
            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(f"Detected {model.config.id2label[label.item()]} with confidence ")
                print(f"{round(score.item(), 3)} at location {box}")
                x1, y1, x2, y2 = box.tolist()
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2,1)
                cv2.putText(frame, model.config.id2label[label.item()], (x1+5, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('model DETR', frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if RUN_LOOP == 1:                                                                                # single file used 
            RUN_LOOP -= 1                                                                                # if it was a file then exit the iteration
    # Release video capture and writer
    if not FLAGS.video.isdigit():     
        cap.release()
    else:
        del cap 
    cv2.destroyAllWindows()        
    writer.release()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
