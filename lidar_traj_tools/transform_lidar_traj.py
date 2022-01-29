from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
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

def save_in_same_dir(file_path, data, ss, field_fmts = '%.6f'):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    np.savetxt(save_file, data, fmt=field_fmts)
    print('save file in: ', save_file)

def rot_trajectory(transformation, traj_data):

    # 四元数到旋转矩阵
    rots = np.zeros(shape=(traj_data.shape[0],4,4))
    for i in range(traj_data.shape[0]):
        i_frame_R = R.from_quat(traj_data[i, 4: 8]).as_matrix()  #3*3
        rots[i] = toRt(i_frame_R, traj_data[i, 1:4])   #4*4

    # for tt in transfrom_list:
    #     rots = np.matmul(tt, rots.T).T
    
    rots = np.matmul(transformation, rots.T).T

    # 旋转矩阵到四元数
    new_traj_data = traj_data.copy()
    for i in range(traj_data.shape[0]):
        new_traj_data[i, 1:4] = rots[i, :3, 3] #平移量
        new_traj_data[i, 4:8] = R.from_matrix(rots[i, :3, :3]).as_quat() #四元数
    return new_traj_data


if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("-F", "--traj_file", type=str, default="/media/data/dyd/humanMoiton/0913/0913daiyudi001_lidar_trajectory.txt")
    parser.add_argument("-R", "--rot_file", type=str, default="/media/data/dyd/humanMoiton/0913/sumRT.txt")
    args = parser.parse_args()
    
    traj_data = np.loadtxt(args.traj_file, dtype=float)
    transformation = np.loadtxt(args.rot_file, dtype=float).reshape(-1, 4, 4)

    # get 4x4 rotation matrix
    sumT = np.eye(4)    
    for T in transformation:
        sumT = np.matmul(T, sumT)
        print(T)
    print('\nSum Rt:\n', sumT)

    if transformation.shape[0] > 1:
        save_rt_path = os.path.join(os.path.dirname(args.rot_file), 'final_transformation.txt')
        np.savetxt(save_rt_path, sumT, fmt='%.6f')

    # rotate traj
    new_traj_data = rot_trajectory(sumT, traj_data)

    # save new traj
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    save_in_same_dir(args.traj_file, new_traj_data, '_transformed', field_fmts = field_fmts)
