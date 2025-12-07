#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Use facebook detectron tracker on either NVIDIA csi or usb camera input or a video file ot single picture file
#
# ref:- Use FB detectron NVIDIA ModelZoo(https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
#
# requires NVIDIA library as below
# git clone https://github.com/NVIDIA-AI-IOT/jetcam
# cd jetcam
# sudo python3 setup.py install
#
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import os, json, cv2, random
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from PIL import Image

# use NVIDA cameras
# git clone https://github.com/NVIDIA-AI-IOT/jetcam
# cd jetcam
# sudo python3 setup.py install
#
import sys
import cv2
from .camera import Camera                                           # from jetcam library on github
import atexit
import numpy as np
import threading
import traitlets
import os
import signal

RUN_LOOP=True
def sig_hangup_handler(sig, frame):
    sys.stderr.write('sig_hangup_handler({})\n'.format(sig))
    try:
        sys.stderr.write('restarting...\n')
        os.execve('/usr/bin/python3', ['/usr/bin/python3', 'fbdetectron_nvdia.py'], {})
    except OSError as e:
        sys.stderr.write("execve():{}\n".format(e))
        os._exit(1)

def handler(signum, frame):
    global RUN_LOOP
    print(f'handler active im stopping.... (signum={signum})')
    RUN_LOOP = False

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

    def _gst_str(self):
        return 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink'.format(self.capture_device, self.capture_width, self.capture_height, self.capture_fps)
    def _read(self):
        re, image = self.cap.read()
        if re:
            image_resized = cv2.resize(image,(int(self.width),int(self.height)))
            return image_resized
        else:
            raise RuntimeError('Could not read image from camera')


# my file is from my custom dataset labelled in robodlow as drone2.v3i.coco/train/_annotations.coco.json
MYFILE="drone2.v3i.coco/train/_annotations.coco.json"
MYDIR="drone2.v3i.coco/train"
DATASET_NAME="drone2"
register_coco_instances(DATASET_NAME, {}, MYFILE, MYDIR)
signal.signal(signal.SIGHUP, sig_hangup_handler)                      # try re-start on a hang-up
signal.signal(signal.SIGUSR1, handler)                                # user graceful kill
signal.signal(signal.SIGUSR2, handler)
if len(sys.argv[0]) >= 1:
    if sys.argv[1] == "csi":
        cap = CSICamera(width=224, height=224, capture_width=1080, capture_height=720, capture_fps=30)
    elif sys.argv[1] == "usb":
        if len(sys.argv[0]) >= 2:
            cap = USBCamera(capture_device=int(sys.argv[2]))        
        else:
            cap = USBCamera(capture_device=1)
    elif sys.argv[1] == "vid":                                                                         # video file input 
        if len(sys.argv[0]) >= 2:
            cap = cv2.VideoCapture(sys.argv[2]) 
        else:
            print("you must specify the video file as 2nd arg")
            return 
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (DATASET_NAME,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025                                                                              # learning rate
cfg.SOLVER.MAX_ITER = 300                                                                                 # max iter
cfg.SOLVER.STEPS = []                                                                                     # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128                                                            # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1                                                                       # number of classes
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))    # use model zoo
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, MYDIR+"model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
drone_metadata = MetadataCatalog.get(DATASET_NAME)
imgPath = "/home/mark/drones_pic.jpg"                                                                       # default test file

while RUN_LOOP == True:
    if len(sys.argv[0]) >= 1:
        if sys.argv[1] == "csi" or sys.argv[1] == "usb" or sys.argv[1] == "vid":
            im = cap.read()
        elif sys.argv[1] == "img":                                                                         # single image not camera 
            if len(sys.argv[0]) >= 2:
                im = cv2.imread(sys.argv[2])
            else:
                print("you must specify the image name as the 2nd arg")
                return
            RUN_LOOP = False           
        else:
            im = cv2.imread(imgPath)  
            RUN_LOOP = False        
    outputs = predictor(im)                                                                               # test the image with teh model
    v = Visualizer(im[:, :, ::-1], metadata=drone_metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("plan",v.get_image()[:, :, ::-1])
    #result = cv2.imwrite("picture-gray.jpg",v.get_image()[:, :, ::-1])
    classes_list= MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    objects = []
    for id in outputs["instances"]._fields["pred_classes"].to('cpu').detach().numpy().copy():
        obj = classes_list[id]
        objects.append(obj)
    object_est = [(k,i) for k,i in zip(objects,outputs["instances"]._fields["scores"].detach().numpy().copy())]
    print(object_est)
    bounding_boxes = outputs["instances"]._fields["pred_boxes"].tensor.cpu().numpy()                                        # bounding box
    img_copy = im
    for i in range(len(bounding_boxes)):
        left_pt = tuple(bounding_boxes[i][0:2].detach().numpy().copy())
        right_pt = tuple(bounding_boxes[i][2:4].detach().numpy().copy())
        cv2.putText(img_copy, f'{i}', left_pt, cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.rectangle(img_copy,left_pt,right_pt,(0,0,155),1)
    cv2.imshow('image',img_copy)
    object_estimation = object_est
    dct = {imgPath:object_estimation}
    bool_array = outputs["instances"]._fields["pred_masks"]
    for j,b_array in enumerate(bool_array):
        array = b_array.cpu().numpy()
        inv_bool_array = []
        for l in array:
            inv_b = []
            for b in l:
                if b == False:
                    inv_b.append(True)
                else:
                    inv_b.append(False)
            inv_bool_array.append(inv_b)
        copy_img = im.copy()
        copy_img[inv_bool_array] = [128,128,128]
        print(type(copy_img))
        out = Image.fromarray(copy_img)
        print(out.mode)
        out.save('saved_data.jpg')
    # We can use `Visualizer` to draw the predictions on the image.
    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # image[:, :, ::-1]はRGBをBGRへ変換している
    #cv2.imshow(out.get_image()[:, :, ::-1])
    #result2 = cv2.imwrite("result.jpg",out.get_image()[:, :, ::-1])
