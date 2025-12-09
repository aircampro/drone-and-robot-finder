#!/usr/bin/python
#
# test an image using curl to the roboflow dataset for observing drones and robots
#
import sys
import shlex
from subprocess import Popen, PIPE, STDOUT
import json

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
    if len(sys.argv[0]) >= 1:
        YOUR_IMAGE=str(sys.argv[1])
    else:
        YOUR_IMAGE="Capture1.jpg"
    # use shell wrapper
    shl = Shell()
    (return_code, stdout) = shl.cmd(f'base64 {YOUR_IMAGE}').pipe('curl -d @- https://detect.roboflow.com/drone2-epzxe/7?api_key=Mqvor12zsKc38eyJUP97').commit()
    if return_code == 0:
        s=stdout.decode('utf-8').split('\n')[3]
        j=json.loads(s)
        print(f"predictions {j['predictions']}")
        print(f"image {j['image']}")
    # call bash script
    p, err = subprocess.Popen(['/bin/bash', './run_curl.sh', f'{YOUR_IMAGE}'], stdout=subprocess.PIPE).communicate().decode('utf-8')
    if err == None:
        jj=json.loads(p)
        print(f"predictions {jj['predictions']}")
        print(f"image {jj['image']}")

