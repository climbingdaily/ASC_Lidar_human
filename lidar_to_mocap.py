from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
from bvh_tool import Bvh
import sys
import os

def toRt(r, t):
    '''
    将3*3的R转成4*4的R
    '''
    share_vector = np.array([0,0,0,1], dtype=float)[np.newaxis, :]
    r = np.concatenate((r, t.reshape(-1,1)), axis = 1)
    r = np.concatenate((r, share_vector), axis=0)
    return r

def save_in_same_dir(file_path, data, ss):
    '''
    将3*3的R转成4*4的R
    '''
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    np.savetxt(save_file, data, fmt='%.6f')

offset_file = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021上科大交流\数据集制作\\LiDAR到hip平移向量.txt"
offset = np.loadtxt(offset_file, dtype=float) # GT轨迹点到hip的平移向量

# 读取文件
if len(sys.argv) == 3:
    lidar_traj = sys.argv[1]
    mocap_file = sys.argv[2]
else:
    lidar_traj = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021上科大交流\\演示数据\\0712\\1st_data_400_to_end\\Frame_xyzRxRyRzTime.txt"
    mocap_file = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021上科大交流\\演示数据\\0712\\test_0712_ASC_corridor.bvh"

m = 409
n = 189

lidar = np.loadtxt(lidar_traj, dtype=float)
with open(mocap_file) as f:
    mocap = Bvh(f.read())
#读取数据
frame_time = mocap.frame_time
frames = mocap.frames
frames = np.asarray(frames, dtype='float32')

key_lidar = lidar[int(m - lidar[0,0]), 1:8] #取第m帧轨迹的RT
key_motion = frames[n,:6] #取第n帧mocap的RT

#取第0帧轨迹的R
R_lidar_init = R.from_quat(lidar[0, 4:8]).as_matrix()
R_lidar_init = toRt(R_lidar_init, lidar[0, 1:4])
R_lidar_init_inv = np.linalg.inv(R_lidar_init)
save_in_same_dir(lidar_traj, R_lidar_init_inv, '_first_frame_inv')

#取第m帧轨迹的RT
R_lidar = R.from_quat(key_lidar[3:7]).as_matrix()
R_lidar = toRt(R_lidar, key_lidar[:3])
R_lidar = np.matmul(R_lidar, offset) #乘第一帧的逆,再平移到hip point

#取第n帧mocap的RT
R_mocap = R.from_euler('zxy', key_motion[3:6], degrees=False).as_matrix()
T_mocap = key_motion[:3]/100
R_mocap = toRt(R_mocap, T_mocap)

mocap_to_lidar = np.matmul(np.linalg.inv(R_mocap), R_lidar)
lidar_to_mocap = np.linalg.inv(mocap_to_lidar)

# 保存文件
save_in_same_dir(mocap_file, mocap_to_lidar, '_mocap_to_lidar')
save_in_same_dir(mocap_file, lidar_to_mocap, '_lidar_to_mocap')

print("Pass!!!")

# python lidar_to_mocap.py [lidar_traj] [mocap_file]