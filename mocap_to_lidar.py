import pandas as pd  
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
from bvh_tools.bvh_tool import Bvh
import sys
import os
from smpl.smpl import SMPL
from smpl.skele2smpl import get_pose_from_bvh
from smpl.generate_ply import save_ply

_mocap_init = np.array([
    [-1, 0, 0, 0],
    [0, 0, 1, 0], 
    [0, 1, 0, 0], 
    [0, 0, 0, 1]])
# mocap_init = R.from_matrix(mocap_init[:3,:3])

# mocap的帧率试试lidar的备注
frame_scale = 1 # mocap是100Hz, lidar是20Hz

def toRt(r, t):
    '''
    将3*3的R转成4*4的R
    '''
    share_vector = np.array([0,0,0,1], dtype=float)[np.newaxis, :]
    r = np.concatenate((r, t.reshape(-1,1)), axis = 1)
    r = np.concatenate((r, share_vector), axis=0)
    return r


def save_pose(filepath, poses, skip=100):
    dirname = os.path.dirname(filepath)
    file_name = Path(filepath).stem
    save_file = os.path.join(dirname, file_name + '_cloud.txt')
    shape = poses.shape
    num = np.arange(shape[0])
    pose_num = num.repeat(shape[1]).reshape(shape[0], shape[1], 1)
    poses = np.concatenate((poses, pose_num), -1)  # 添加行号到末尾
    poses_save = poses[num % skip ==0].reshape(-1, 4) #每隔skip帧保存一下
    np.savetxt(save_file, poses_save, fmt='%.6f')
    print('保存pose到: ', save_file)

def save_in_same_csv(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.csv')
    data.to_csv(save_file)
    print('save csv in: ', save_file)

def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('save traj in: ', save_file)

def load_data(lidar_file, rotpath, pospath):
    lidar = np.loadtxt(lidar_file, dtype=float)
    pos_data_csv=pd.read_csv(pospath, dtype=np.float32)
    rot_data_csv=pd.read_csv(rotpath, dtype=np.float32)

    pos_data = np.asarray(pos_data_csv) /100 # cm -> m
    mocap_length = pos_data.shape[0]
    pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
    rot_data = np.asarray(rot_data_csv) # 度
    return lidar, pos_data, rot_data


def get_overlap(lidar, mocap, lidar_key, mocap_key, mocap_frame_time = 0.01, save_file = True):
    # 1. 读取时间戳
    lidar_time = lidar[:, -1]
    mocap_time = mocap[:, 0]
    # mocap_time = mocap_time.astype('float64')
    start = lidar_key - mocap_key

    # 2. 根据lidar的时间戳，选取对应的mocap的帧
    _lidar_time = lidar_time - start
    _lidar = []
    _mocap_id = []
    _mocap = []
    for i, l in enumerate(lidar):
        t = _lidar_time[i]
        index = np.where(abs(mocap_time - t) - mocap_frame_time/2 <= 1e-5)[0]
        if(index.shape[0] > 0):
            _mocap.append(mocap[index[0]])
            _mocap_id.append(index[0])
            _lidar.append(l)
        # else:
        #     print(i)

    # 3. 保存修改的mocap文件 
    _mocap = np.asarray(_mocap)
    _mocap_id = np.asarray(_mocap_id)
    _lidar = np.asarray(_lidar)

    
    if save_file:
        # save_in_same_dir(mocap_root_file, _mocap_root, '_syncTime')    
        save_in_same_dir(lidar_file, _lidar, '_syncTime')    
    return _lidar, _mocap_id

def register_mocap(mocap_sync_id, lidar_sync, rot_data, pos_data, mocap_init = _mocap_init):
    lidar_first = R.from_quat(lidar_sync[0, 4: 8]).as_matrix() #第一帧的矩阵
    mocap_first = R.from_euler('yxz', rot_data[mocap_sync_id[0], 1:4], degrees=True).as_matrix() #第一帧的旋转矩阵
    mocap_first = np.matmul(mocap_init[:3,:3], mocap_first) #第一帧的旋转矩阵，乘上 mocap坐标系 -> lidar坐标系的变换矩阵

    sync_shape = lidar_sync.shape[0]
    poses = np.zeros(shape=(sync_shape, pos_data.shape[1], 3))
    new_rot = np.zeros(shape=(sync_shape, rot_data.shape[1]))
    new_pos = np.zeros(shape=(sync_shape, 3))
    for i in range(sync_shape):
        # 读取 i 帧的 RT
        R_lidar = R.from_quat(lidar_sync[i, 4: 8]).as_matrix()  #3*3
        R_lidar = np.matmul(R_lidar, np.linalg.inv(lidar_first))
        R_lidar = toRt(R_lidar, lidar_sync[i, 1:4])   #4*4

        # 读取对应 mocap的hip的rt
        # mocap_number = (i - lidar_key + lidar_start) * frame_scale + mocap_key # 对应的mocap的帧
        mocap_number = mocap_sync_id[i]
        R_mocap = R.from_euler('yxz', rot_data[mocap_number, 1:4], degrees=True).as_matrix() #原始数据
        R_mocap = toRt(R_mocap, pos_data[mocap_number, 0].copy())

        R_mocap = np.matmul(mocap_init, R_mocap) # 变换到lidar坐标系
        R_mocap[:3,:3] = np.matmul(R_mocap[:3, :3], np.linalg.inv(mocap_first)) # 右乘第一帧旋转矩阵的逆

        # 求mocap到Lidar的变换关系
        mocap_to_lidar = np.matmul(R_lidar, np.linalg.inv(R_mocap))

        # 将变换矩阵应用于单帧所有点
        pos_init = np.matmul(mocap_init[:3,:3], pos_data[mocap_number].T) # 3 * m, 先坐标系变换
        poses[i] = np.matmul(mocap_to_lidar[:3,:3], pos_init).T + mocap_to_lidar[:3,3] # m * 3，再进行旋转平移

        # 将mocap的所有关节的旋转都改变
        new_rot[i] = rot_data[mocap_number]
        new_pos[i] = pos_data[mocap_number, 0]
        # new_rot[i, 0] = rot_data[mocap_number, 0].copy()
        # for j in range(rot_data.shape[1]//3):
        #     R_ij = R.from_euler(
        #         'yxz', rot_data[mocap_number, j*3 + 1:j*3 + 4], degrees=True).as_matrix()
      
            # R_ijj = np.matmul(mocap_init[:3,:3], R_ij)  
            # R_ijj = np.matmul(mocap_to_lidar[:3,:3], R_ijj) # mocap->lidar 配准旋转矩阵
            # R_ijj = np.matmul(mocap_init[:3,:3], R_ijj)  
            # new_rot[i, j*3 + 1:j*3 + 4] = R.from_matrix(R_ijj).as_euler('yxz', degrees=True)
        return new_pos, new_rot, poses

def rot_to_smpl(rotpath, new_rot, lidar_sync, mocap_init = _mocap_init):
    import torch
    from tqdm import tqdm
    savedir = os.path.join(os.path.dirname(rotpath), 'SMPL')
    rot_data_csv=pd.read_csv(rotpath, dtype=np.float64)
    new_rot_csv = pd.DataFrame(new_rot, columns = [col for col in rot_data_csv.columns])
    
    os.makedirs(savedir, exist_ok=True)
    smpl_out_dir = os.path.join(savedir, Path(rotpath).stem)
    os.makedirs(smpl_out_dir, exist_ok=True)
    smpl = SMPL()
    sync_shape = new_rot.shape[0]
    bar = tqdm(range(sync_shape))

    for count in bar:
        vertices = smpl(torch.from_numpy(get_pose_from_bvh(
            new_rot_csv, count, False)).unsqueeze(0).float(), torch.zeros((1, 10)))
        vertices = vertices.squeeze().cpu().numpy()

        translation = lidar_sync[count, 1:4]
        rot = R.from_quat(lidar_sync[count, 4: 8]).as_matrix()
        if count == 0:
            rot_0 = R.from_quat(lidar_sync[0, 4: 8]).as_matrix()
            mocap_pos = np.matmul(mocap_init[:3, :3], new_pos[0])
            m_to_l = mocap_pos - translation
            m_to_l = np.matmul(np.linalg.inv(rot_0), m_to_l)
        vertices = np.matmul(mocap_init[:3, :3], vertices.T)
        # vertices = np.matmul(mocap_to_lidar[:3, :3], vertices).T + translation
        vertices = vertices.T + translation + np.matmul(rot, m_to_l) # 初始化的偏移量需要乘上lidar的旋转矩阵
        # vertices = vertices.T  + mocap_pos # 直接使用mocap的轨迹
        
        ply_save_path = os.path.join(smpl_out_dir, str(count) + '_smpl.ply')
        save_ply(vertices,ply_save_path)
        bar.set_description("Save number %d/%d ply in " % (count, lidar_sync.shape[0]))
    print('SMPL saved in: ', smpl_out_dir)
    return new_rot_csv

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('请输入 [*pos.csv] [*rot.csv] [*lidar_traj.txt]')
        pospath = "e:\\SCSC_DATA\HumanMotion\\0913\\001\\mocap\\0913daiyudi001_pos.csv"
        rotpath = "e:\\SCSC_DATA\HumanMotion\\0913\\001\\mocap\\0913daiyudi001_rot.csv"
        lidar_file = "e:\\SCSC_DATA\HumanMotion\\0913\\001\\lidar\\lidar_trajectory_afterFit.txt"
        # in_file = 'sys.argv[4]'
        # ex_file = 'sys.argv[5]'
        # dist_file = 'sys.argv[5]'
    else:
        pospath = sys.argv[1]
        rotpath = sys.argv[2]
        lidar_file = sys.argv[3]
        # in_file = sys.argv[4]
        # ex_file = sys.argv[5]
        # dist_file = sys.argv[5]

    # K_IN = np.loadtxt(in_file, dtype=float)
    # K_EX = np.loadtxt(ex_file, dtype=float)
    # dist_coeff = np.loadtxt(dist_file, dtype=float)

    # 1. 读取数据
    lidar, pos_data, rot_data = load_data(lidar_file, rotpath, pospath)
    
    # 2. 输入lidar中对应的帧和mocap中对应的帧, 求得两段轨迹中的公共部分
    lidar_key = 351.666     # 时间同步帧
    mocap_key = 23.04    # 时间同步帧
    lidar_sync, mocap_sync_id = get_overlap(lidar, rot_data, lidar_key, mocap_key)

    # 3. 将mocap配准到lidar，得到RT，应用于该帧的所有点
    new_pos, new_rot, poses = register_mocap(mocap_sync_id, lidar_sync, rot_data, pos_data)

    # 4. 转换SMPL 
    new_rot_csv = rot_to_smpl(rotpath, new_rot, lidar_sync)

    # 4. 保存pose
    save_in_same_csv(rotpath, new_rot_csv, '_trans_RT')
    # save_in_same_dir(lidar_file, lidar_sync, '_与mocap重叠部分') #保存有效轨迹
    # save_pose(pospath, position, skip = 40)

