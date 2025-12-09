#!/usr/bin/python
#
# Test an image using curl to the roboflow cloud model specified for observing drones and robots and markup the picture with the observation returned from the server
#
import sys
import shlex
from subprocess import Popen, PIPE, STDOUT
import json
import cv2

# this is the list of classes from the export yolov11 data.yaml file
class_names = ['balloom', 'drone', 'gyro', 'person', 'plane', 'robot']
# defines the list of colors for each class
class_colors = {'balloom' : (127, 127, 0), 'drone' : (127, 0, 127), 'gyro' : (0, 127, 127), 'person' : (255, 0, 0), 'plane' : (0, 255, 0), 'robot' : (0, 0, 255)}
# class for bbox co-ordinates 
class BoundingBox():
    def __init__(self, x, y, w, h, classi):
        self.x  = x                                                           # start co-ord
        self.y  = y
        self.w  = w                                                           # width
        self.h  = h                                                           # height
        self.x1 = x + w                                                       # end co-ord
        self.y1 = y + h
        self.classid = classi                                                 # classification of object
        self.xc = x + (w/2)                                                   # center co-ord
        self.yc = y + (h/2)
    def movement(self, pos, arr):                                             # for video (multiple frames) calculate the movement between frames (optical flow)
        self.ox = arr[pos].x - self.x
        self.oy = arr[pos].y  - self.y
        self.ox1 = arr[pos].x1 - self.x1
        self.ox1 = arr[pos].y1 - self.y1 
        self.oxc = arr[pos].xc - self.xc
        self.oxc = arr[pos].yc - self.yc 
  
# this class wraps the subprocess for shell and pipe
# 
class Shell(object):
    def __init__(self):
        self.p  = None
    def cmd(self, cmd):
        p_args = {'stdin'     : None,
                  'stdout'    : PIPE,
                  'stderr'    : STDOUT,
                  'shell'     : False,}
        return self._execute(cmd, p_args)
    def pipe(self, cmd):
        p_args = {'stdin'     : self.p.stdout,
                  'stdout'    : PIPE,
                  'stderr'    : STDOUT,
                  'shell'     : False,}
        return self._execute(cmd, p_args)
    def _execute(self, cmd, p_args):
        try :
            self.p = Popen(shlex.split(cmd), **p_args)
            return self
        except :
            print('Command Not Found: %s' % (cmd))
    def commit(self):
        result = self.p.communicate()[0]
        status = self.p.returncode
        return (status, result)

if __name__ == '__main__':
    THRESHOLD = 50.0                                                                                                                # threshold to draw images 
    if len(sys.argv[0]) >= 1:
        YOUR_IMAGE=str(sys.argv[1])
        if len(sys.argv[0]) >= 2:
            YOUR_IMAGE2=str(sys.argv[2])
        else:
            YOUR_IMAGE2=None 
       if len(sys.argv[0]) >= 3:  
           THRESHOLD = int(sys.argv[3])       
    else:
        YOUR_IMAGE="Capture1.jpg"
        YOUR_IMAGE2=None
    frame = cv2.imread(YOUR_IMAGE)
    shl = Shell()
    (return_code, stdout) = shl.cmd(f'base64 {YOUR_IMAGE}').pipe('curl -d @- https://detect.roboflow.com/drone2-epzxe/7?api_key=Mqvor12zsKc38eyJUP97').commit()
    if return_code == 0:
        s=stdout.decode('utf-8').split('\n')[3]
        j=json.loads(s)
        print(f"predictions {j['predictions']}")
        print(f"image {j['image']}")
        preds = j['predictions']
        vec_preds = []
        for i in range(0, len(preds)):
            print(f"x {j['predictions'][i]['x']} : y {j['predictions'][i]['y']} : w {j['predictions'][i]['width']} : h {j['predictions'][i]['height']} : {j['predictions'][i]['class']}")  
            bbox = BoundingBox(j['predictions'][i]['x'], j['predictions'][i]['y'], j['predictions'][i]['width'], j['predictions'][i]['height'],j['predictions'][i]['class'])   
            try:            
                cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x1, bbox.y1), class_colors[bbox.classid], 2) 
                text = f"{bbox.classid}"            
                cv2.rectangle(frame, (bbox.x - 1, bbox.y - 20), (bbox.x + len(text) * 12, bbox.y), class_colors[bbox.classid], -1)
                cv2.putText(frame, text, (bbox.x + 5, bbox.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except:
                cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x1, bbox.y1), (255,255,255), 2) 
                text = f"{bbox.classid}"            
                cv2.rectangle(frame, (bbox.x - 1, bbox.y - 20), (bbox.x + len(text) * 12, bbox.y), (255,255,255), -1)
                cv2.putText(frame, text, (bbox.x + 5, bbox.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)            
            vec_preds.append(bbox)
        cv2.imshow('Drone Robot People Detector', frame) 
    if not YOUR_IMAGE2 == None:                                                                                            # if we are comparing 2 frames for optical flow between observed classes      
        frame = cv2.imread(YOUR_IMAGE2)
        (return_code, stdout) = shl.cmd(f'base64 {YOUR_IMAGE2}').pipe('curl -d @- https://detect.roboflow.com/drone2-epzxe/7?api_key=Mqvor12zsKc38eyJUP97').commit()
        if return_code == 0:
            s=stdout.decode('utf-8').split('\n')[3]
            j=json.loads(s)
            print(f"predictions {j['predictions']}")
            print(f"image {j['image']}")
            preds = j['predictions']
            for i in range(0, len(preds)):
                print(f"x {j['predictions'][i]['x']} : y {j['predictions'][i]['y']} : w {j['predictions'][i]['width']} : h {j['predictions'][i]['height']} : {j['predictions'][i]['class']}")  
                bbox = BoundingBox(j['predictions'][i]['x'], j['predictions'][i]['y'], j['predictions'][i]['width'], j['predictions'][i]['height'],j['predictions'][i]['class'])   
                if len(preds) == len(vec_preds):                                                                           # should really check unique id but as we dont have it ill check for asme number of classes
                    bbox.movement(i, vec_preds)                                                                            # calculate the movement between the frames for each class 
                    text = f"{bbox.classid} x:{bbox.ox} y:{bbox.oy}"  
                else:
                    text = f"{bbox.classid}"  
                if j['predictions'][i]['confidence'] >= THRESHOLD:                    
                    try:            
                        cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x1, bbox.y1), class_colors[bbox.classid], 2)           
                        cv2.rectangle(frame, (bbox.x - 1, bbox.y - 20), (bbox.x + len(text) * 12, bbox.y), class_colors[bbox.classid], -1)
                        cv2.putText(frame, text, (bbox.x + 5, bbox.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    except:
                        cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x1, bbox.y1), (255,255,255), 2)          
                        cv2.rectangle(frame, (bbox.x - 1, bbox.y - 20), (bbox.x + len(text) * 12, bbox.y), (255,255,255), -1)
                        cv2.putText(frame, text, (bbox.x + 5, bbox.y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)            
            cv2.imshow('DrRoPe Detector 2nd Frame Comparison', frame)        
