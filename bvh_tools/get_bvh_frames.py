from pathlib import Path
import numpy as np
import sys
import os
from bvh_tool import Bvh
from scipy.spatial.transform import Rotation as R

# 读取文件
if len(sys.argv) == 3:
    joints_file = sys.argv[1]
    frame_num = int(sys.argv[2])
else:
    print('please input [bvh file path] and [frame number]')
    joints_file = "e:\\SCSC_DATA\HumanMotion\\0828\\haiyunyuandaiyudi002_daiyudi.bvh"
    frame_num = 10

print(joints_file)
with open(joints_file) as f:
    mocap = Bvh(f.read())


#读取数据
frame_time = mocap.frame_time
frames = mocap.frames
frames = np.asarray(frames, dtype='float32')

header = mocap.data.split('MOTION')[0]
frames = mocap.frames[:frame_num]


# 保存文件
dirname = os.path.dirname(joints_file)
# file_name = os.path.basename(joints_file)
file_name = Path(joints_file).stem
save_file = os.path.join(dirname, file_name + '_' + str(frame_num) + '.bvh')

with open(save_file, 'w') as f:
    f.write(header)
    f.write('MOTION\n')
    f.write('Frames: ' + str(frame_num) + '\n')
    f.write('Frame Time: ' + str(frame_time) + '\n')
    for i, frame in enumerate(frames):
        if i > frame_num:
            break
        for s in frame:
            f.write(s + ' ')
        f.write('\n')