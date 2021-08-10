import open3d as o3d
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd  
from bvh_tool import Bvh

def triangle_pcd(start, end):
    '''
    定义三角形的点云
    :return:
    '''
    triangle_points = np.concatenate((start.reshape(1,3), end),axis=0)
    # triangle_points = np.array(
    #     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    lines = [[0, 1], [0, 2], [0, 3]]  # Right leg
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Default blue
    # 定义三角形的三个角点
    point_pcd = o3d.geometry.PointCloud()  # 定义点云
    point_pcd.points = o3d.utility.Vector3dVector(triangle_points)
 
    # 定义三角形三条连接线
    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(colors)
    line_pcd.points = o3d.utility.Vector3dVector(triangle_points)
    return line_pcd, point_pcd
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        key = '-l'
        lidar_file = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\02lidar\\traj_with_timestamp_变换后的轨迹.txt"
    else:
        key = sys.argv[1]
        if key == '-l':
            lidar_file = sys.argv[2]
        elif key == '-b':
            mocap_file = sys.argv[2]
        elif key == '-c':
            pos_file = sys.argv[2]
            rot_file = sys.argv[3]
        else:
            print('python visualize_RT.py [-l] [lidar_traj_path]')
            print('python visualize_RT.py [-b] [bvh_path]')
            print('python visualize_RT.py [-c] [csv_pos_path] [csv_rot_path]')
            exit()
    geometies = []
    pcdroot = 'E:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\实验室6楼'
    pcd_floor = o3d.io.read_point_cloud(os.path.join(pcdroot, '地板.pcd'))
    pcd_wall1 = o3d.io.read_point_cloud(os.path.join(pcdroot, '墙1.pcd'))
    pcd_wall2 = o3d.io.read_point_cloud(os.path.join(pcdroot, '墙2.pcd'))
    geometies.append(pcd_floor)
    geometies.append(pcd_wall1)
    geometies.append(pcd_wall2)

    if key == '-l':
        rt_file = np.loadtxt(lidar_file, dtype=float)        
        R_init = R.from_quat(rt_file[0, 4: 8]).as_matrix()  #3*3
        for i in range(0, rt_file.shape[0], 20):
            # 读取 i 帧的 RT
            R_lidar = R.from_quat(rt_file[i, 4: 8]).as_matrix()  #3*3
            R_lidar = np.matmul(R_lidar, np.linalg.inv(R_init))
            R_T = rt_file[i, 1:4].reshape(1,3)   #1*3
            R_lidar = R_lidar.T + R_T
            line_pcd, point_pcd = triangle_pcd(R_T, R_lidar)
            geometies.append(line_pcd)
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

    o3d.visualization.draw_geometries(geometry_list  = geometies, window_name = 'Draw RT')
    # # 绘制open3d坐标系
    # line_set = o3d.geometry.LineSet()
    # point_cloud = o3d.geometry.PointCloud()
    # axis_pcd = o3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # # 在3D坐标上绘制点：坐标点[x,y,z]对应R，G，B颜色
    # points = np.array([[1, 0, 0]], dtype=np.float64)
    # colors = [[1, 0, 0]]
 
    # # 方法1（非阻塞显示）
    # vis = o3d.visualization
    # vis.create_window(window_name='Open3D_1', width=600, height=600, left=10, top=10, visible=True)
    # vis.get_render_option().point_size = 10  # 设置点的大小
    # # 先把点云对象添加给Visualizer
    # vis.add_geometry(axis_pcd)
 
    # line_pcd, point_pcd = triangle_pcd()
    # geometies = []
    # geometies.append(line_pcd)
    # geometies.append(point_pcd)
 
