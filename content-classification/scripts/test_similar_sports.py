"""
Utility script to test the model on a private set of videos of sports similar to football with scenedetection script https://github.com/livepeer/lpms/blob/master/cmd/scenedetection/scenedetection.go
Videos are not included into the dataset.
"""

import sys
import glob
import subprocess
import os

res_list = []
for f in sorted(glob.glob(sys.argv[1] + '/*.*')):
    args = f"./scenedetection 0 'p' P144p30fps16x9 nv 0".split()
    args[2] = f'"{f}"'
    p = subprocess.Popen(args,
                         shell=False, stdout=subprocess.PIPE, env=dict(LD_LIBRARY_PATH='/projects/livepeer/src/compiled_debug/lib'))
    p.wait()
    l = p.stdout.readline().decode('utf-8')
    while 'detectdata=' not in l and l is not None:
        l = p.stdout.readline().decode('utf-8')
    print(l)
    res = '"'+os.path.basename(f) + '" ' + l.split(' ')[-1].replace(']', '').replace('1:', '')
    print(res)
    res_list.append(res)
with open('similar_sports.csv', mode='w') as f:
    f.writelines(res_list)

    