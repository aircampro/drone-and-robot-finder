#!/usr/bin/python
#
# test an image using curl to the roboflow dataset for observing drones and robots
#
import sys
from subprocess import run

def bash(command):
    run(command.split())

if __name__ == '__main__':
    if len(sys.argv[0]) >= 1:
        YOUR_IMAGE=str(sys.argv[1])
    else:
        YOUR_IMAGE="Capture1"
    CMD=f"base64 {YOUR_IMAGE}.jpg | curl -d @- \
      \"https://detect.roboflow.com/drone2-epzxe/7?api_key=Mqvor12zsKc38eyJUP97\""
    bash(CMD)