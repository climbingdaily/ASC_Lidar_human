import open3d as o3d
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd  
from bvh_tools.bvh_tool import Bvh
from visualization.Timer import Timer
import json
import pickle as pkl
from o3dvis import o3dvis, Keyword
# sys.path.insert(0, './')
# sys.path.insert(1, '../')

def load_scene(pcd_path, scene_name):
    print('Loading scene...')
    scene_pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, scene_name + '.pcd'))
    # print('Loading normals...')
    # normal_file = os.path.join(pcd_path, scene_name + '_normals.pkl')
    # if os.path.exists(normal_file):
    #     with open(normal_file, 'rb') as f:
    #         normals = pkl.load(f)
    #     scene_pcd.normals = o3d.utility.Vector3dVector(normals)
    # else:
    #     scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=80))
    #     normals = np.asarray(scene_pcd.normals)
    #     with open(normal_file, 'wb') as f:
    #         pkl.dump(normals, f)
    #     print('Save scene normals in: ', normal_file)

    # scene_pcd.voxel_down_sample(voxel_size=0.02)
    print('Scene loaded...')
    return scene_pcd

if __name__ == "__main__":
    
    lidar_file = "E:\\SCSC_DATA\HumanMotion\\1023\\shiyanlou002_lidar_filt_synced_offset.txt"
    plydir = 'E:\\SCSC_DATA\HumanMotion\\visualization\\contact_compare\\\climbing'
    # pcd_dir = 'E:\\SCSC_DATA\\HumanMotion\\scenes'
    pcd_dir = 'J:\\Human_motion\\visualization'
    

    if len(sys.argv) < 2:
        key = '-m'

    else:
        key = sys.argv[1]
        if key == '-l':
            lidar_file = sys.argv[2]
        elif key == '-b':
            mocap_file = sys.argv[2]
        elif key == '-c':
            pos_file = sys.argv[2]
            rot_file = sys.argv[3]
        elif key == '-m':
            plydir = sys.argv[2]
            scene_name = sys.argv[3]
            # lidar_file = sys.argv[3]
        else:
            print('python visualize_RT.py [-l] [lidar_traj_path]')
            print('python visualize_RT.py [-b] [bvh_path]')
            print('python visualize_RT.py [-c] [csv_pos_path] [csv_rot_path]')
            exit()
    geometies = []
    # start_lidar_idx = int(np.loadtxt(lidar_file, dtype=np.float64)[0,0]) 
    # positions = np.loadtxt(lidar_file, dtype=np.float64)[:, 1:4]
    
    # scene_name = 'climbinggym1101'
    # scene_name = 'lab_building'
    # scene_name = 'campus'

    scene_pcd = load_scene(pcd_dir, scene_name)
    vis = o3dvis()
    vis.add_scene_gemony(scene_pcd)

    if key == '-l':
        rt_file = np.loadtxt(lidar_file, dtype=float)        
        R_init = R.from_quat(rt_file[0, 4: 8]).as_matrix()  #3*3
        for i in range(0, rt_file.shape[0], 20):
            # 读取 i 帧的 RT
            R_lidar = R.from_quat(rt_file[i, 4: 8]).as_matrix()  #3*3
            R_lidar = np.matmul(R_lidar, np.linalg.inv(R_init)) # 乘第一帧的逆
            R_T = rt_file[i, 1:4].reshape(1,3)   #1*3
            R_lidar = R_lidar.T + R_T
            line_pcd, point_pcd = triangle_pcd(R_T, R_lidar)
            # geometies.append(line_pcd)
            vis.add_geometry(line_pcd)
            # geometies.append(point_pcd)

    elif key == '-b':
        mocap_init = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0,]])


        with open(mocap_file) as f:
            mocap = Bvh(f.read())
        frame_time = mocap.frame_time
        frames = mocap.frames
        frames = np.asarray(frames, dtype='float32')
        R_init = R.from_euler('yxz', frames[0, 3:6], degrees=True).as_matrix()
        R_init = np.matmul(mocap_init, R_init)

        for i in range(0, frames.shape[0], 100):
            R_mocap = R.from_euler('yxz', frames[i, 3:6], degrees=True).as_matrix()
            T_mocap =frames[i, :3] / 100    # cm -> m

            R_mocap = np.matmul(mocap_init, R_mocap)
            T_mocap = np.matmul(mocap_init, T_mocap)
            R_mocap = np.matmul(R_mocap, np.linalg.inv(R_init))
        
            R_mocap = R_mocap.T + T_mocap
            line_pcd, point_pcd = triangle_pcd(T_mocap, R_mocap)
            geometies.append(line_pcd)

    elif key == '-c':
        mocap_init = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0,]])

        pos_data=pd.read_csv(pos_file, dtype=np.float32)
        rot_data=pd.read_csv(rot_file, dtype=np.float32)
        pos_data = np.asarray(pos_data) /100 # cm -> m
        mocap_length = pos_data.shape[0]

        pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
        rot_data = np.asarray(rot_data) # 度

        R_init = R.from_euler('yxz', rot_data[0, 1:4], degrees=True).as_matrix()
        R_init = np.matmul(mocap_init, R_init)
        for i in range(0, mocap_length, 100):
            R_mocap = R.from_euler('yxz', rot_data[i, 1:4], degrees=True).as_matrix()
            T_mocap = pos_data[i, 0].copy()
            
            R_mocap = np.matmul(mocap_init, R_mocap)
            T_mocap = np.matmul(mocap_init, T_mocap)

            R_mocap = np.matmul(R_mocap, np.linalg.inv(R_init))
            R_mocap = R_mocap.T + T_mocap
            line_pcd, point_pcd = triangle_pcd(T_mocap, R_mocap)

            geometies.append(line_pcd)
    
    elif key == '-m':
        with open('.\\vertices\\all_new.json') as f:
            all_vertices = json.load(f)
        back = all_vertices['back_new']
        left_heel = all_vertices['left_heel']
        left_toe = all_vertices['left_toe']
        right_heel = all_vertices['right_heel']
        right_toe = all_vertices['right_toe']
        
    mesh_list = []
    sphere_list = []

    while True:
        with Timer('update renderer', True):
            # o3dcallback()
            vis.waitKey(10, helps=False)
            if Keyword.READ:
                # 读取mesh文件
                meshfiles = os.listdir(plydir)
                for mesh in mesh_list:
                    vis.remove_geometry(mesh, reset_bounding_box = False)
                mesh_list.clear()
                mesh_l1 = []    # 无优化的结果
                mesh_l2 = []    # 动捕的原始结果
                mesh_l3 = []    # 优化的结果
                mesh_l4 = []    # 其他类型
                for plyfile in meshfiles:
                    if plyfile.split('.')[-1] != 'ply':
                        continue
                    # print('name', plyfile)
                    if plyfile.split('_')[-1] == 'opt.ply':
                        mesh_l1.append(plyfile)
                    elif plyfile.split('_')[-1] == 'mocap.ply':
                        mesh_l2.append(plyfile)
                    elif plyfile.split('_')[-1] == 'smpl.ply':
                        mesh_l3.append(plyfile)
                    else:
                        mesh_l4.append(plyfile)

                with open('J:\\Human_motion\\visualization\\info_video_climbing.json') as f:
                    info = json.load(f)['climbing_top_2']

                mesh_list += vis.add_mesh_by_order(plydir, mesh_l1, 'red',
                                               'wo_opt_' + info['name'], 
                                               start=info['start'], end=info['end'], info=info)
                mesh_list += vis.add_mesh_by_order(plydir, mesh_l2, 'yellow',
                                               'mocap_' + info['name'], 
                                               start=info['start'], end=info['end'], info=info)
                mesh_list += vis.add_mesh_by_order(plydir, mesh_l3, 'blue',
                                               'with_opt_' + info['name'], 
                                               start=info['start'], end=info['end'], info=info)
                mesh_list += vis.add_mesh_by_order(plydir, mesh_l4,
                                               'blue', 'others' + info['name'], order=False)

                Keyword.READ = False

            sphere_list = vis.visualize_traj(plydir, sphere_list)