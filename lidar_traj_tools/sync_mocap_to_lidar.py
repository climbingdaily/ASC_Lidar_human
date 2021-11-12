from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import sys
import os
import json

def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('save traj in: ', save_file)


def get_overlap(lidar_file, mocap_root_file, 
                lidar_sync, mocap_sync, mocap_frame_time, save_file=True):
    # 1. 读取数据
    lidar = np.loadtxt(lidar_file, dtype=float)
    mocap_root = np.loadtxt(mocap_root_file, dtype=float)

    # 2. 读取时间戳
    lidar_time = lidar[:,-1]
    mocap_time = mocap_root[:,-1]
    start = lidar_sync - mocap_sync

    # 3. 根据lidar的时间戳，选取对应的mocap的帧
    _lidar_time = lidar_time - start
    _lidar = []
    _mocap_root = []
    for i, l in enumerate(lidar):
        t = _lidar_time[i]
        index = np.where(abs(mocap_time - t) <= mocap_frame_time/2)[0]
        if(index.shape[0] > 0):
            _mocap_root.append(mocap_root[index[0]])
            _lidar.append(l)

    # 4. 保存修改的mocap文件 
    _mocap_root = np.asarray(_mocap_root)
    _lidar = np.asarray(_lidar)
    if save_file:
        save_in_same_dir(mocap_root_file, _mocap_root, '_synced')    
        save_in_same_dir(lidar_file, _lidar, '_synced')    
    return _mocap_root, _lidar

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('请输入 [*lidar_traj.txt] [*mocap_root.txt]')
        lidar_file = "e:\\SCSC_DATA\HumanMotion\\1009\\001\\lidar_trajectory.txt"
        mocap_root_file = "e:\\SCSC_DATA\HumanMotion\\1009\\001\\1009shiyanlou_001_root.txt"
    else:
        lidar_file = sys.argv[1]
        mocap_root_file = sys.argv[2]

    init_json = os.path.join(os.path.dirname(mocap_root_file), 'init.json')
    with open(init_json) as init:
        init_params = json.load(init)
        
    mocap_sync = float(init_params['mocap_sync'])
    lidar_sync = float(init_params['lidar_sync'])
    mocap_framerate = int(init_params['mocap_framerate'])
    
    mocap_root, lidar= get_overlap(lidar_file, mocap_root_file, lidar_sync, mocap_sync, 1/mocap_framerate)
