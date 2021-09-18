import open3d as o3d
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd  
from bvh_tools.bvh_tool import Bvh
from visualization.Timer import Timer
# sys.path.insert(0, './')
# sys.path.insert(1, '../')

rotate = False
def o3d_callback_rotate():
    global rotate
    rotate = not rotate
    return False

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh

# vis = o3d.visualization.Visualizer()
vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.register_key_callback(ord('A'), o3d_callback_rotate)
vis.create_window(window_name='RT', width=2096, height=1024)

def toRt(r, t):
    '''
    将3*3的R转成4*4的R
    '''
    share_vector = np.array([0,0,0,1], dtype=float)[np.newaxis, :]
    r = np.concatenate((r, t.reshape(-1,1)), axis = 1)
    r = np.concatenate((r, share_vector), axis=0)
    return r

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

camera = {
  'phi': 0,
  'theta': -30,
  'cx': 0.,
  'cy': 0.5,
  'cz': 3.}


def init_camera(camera_pose):
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    init_param.extrinsic = np.array(camera_pose)
    ctr.convert_from_pinhole_camera_parameters(init_param)

def set_camera(camera_pose):
    theta, phi = np.deg2rad(-(camera['theta'] + 90)), np.deg2rad(camera['phi'] + 180)
    theta = theta + np.pi
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    rot_x = np.array([
        [1., 0., 0.],
        [0., ct, -st],
        [0, st, ct]
    ])
    rot_z = np.array([
        [cp, -sp, 0],
        [sp, cp, 0.],
        [0., 0., 1.]
    ])
    camera_pose[:3, :3] = rot_x @ rot_z
    return camera_pose

def get_camera():
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    return np.array(init_param.extrinsic)

def o3dcallback(camera_pose=None):
    # if rotate:
    # camera['phi'] += np.pi/10
    # camera_pose = set_camera(get_camera())
    # camera_pose = np.array([[-0.927565, 0.36788, 0.065483, -1.18345],
    #                         [0.0171979, 0.217091, -0.976, -0.0448631],
    #                         [-0.373267, -0.904177, -0.207693, 8.36933],
    #                         [0, 0, 0, 1]])
    print(camera_pose)
    init_camera(camera_pose)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        key = '-l'
        lidar_file = "E:\\SCSC_DATA\HumanMotion\\0828\\20210828haiyunyuan002_pcap_to_txt_0_to_1600\\traj_with_timestamp_变换后的轨迹.txt"
        plydir = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\mocap_csv\SMPL\\02_with_lidar_rot"
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
            # lidar_file = sys.argv[3]
        else:
            print('python visualize_RT.py [-l] [lidar_traj_path]')
            print('python visualize_RT.py [-b] [bvh_path]')
            print('python visualize_RT.py [-c] [csv_pos_path] [csv_rot_path]')
            exit()
    geometies = []
    pcdroot = 'E:\\SCSC_DATA\\HumanMotion\\0913'
    pcd_floor = o3d.io.read_point_cloud(os.path.join(pcdroot, 'colormap.pcd'))
    # pcd_wall1 = o3d.io.read_point_cloud(os.path.join(pcdroot, '墙1.pcd'))
    # pcd_wall2 = o3d.io.read_point_cloud(os.path.join(pcdroot, '墙2.pcd'))
    # pcd_room = o3d.io.read_point_cloud(os.path.join(pcdroot, '硕士间.pcd'))
    geometies.append(pcd_floor)
    # geometies.append(pcd_wall1)
    # geometies.append(pcd_wall2)
    # geometies.append(pcd_room)

    for g in geometies:
        vis.add_geometry(g)
    if key == '-l':
        rt_file = np.loadtxt(lidar_file, dtype=float)        
        R_init = R.from_quat(rt_file[0, 4: 8]).as_matrix()  #3*3
        for i in range(0, rt_file.shape[0], 20):
            # 读取 i 帧的 RT
            R_lidar = R.from_quat(rt_file[i, 4: 8]).as_matrix()  #3*3
            # R_lidar = np.matmul(R_lidar, np.linalg.inv(R_init)) # 乘第一帧的逆
            R_T = rt_file[i, 1:4].reshape(1,3)   #1*3
            R_lidar = R_lidar.T + R_T
            line_pcd, point_pcd = triangle_pcd(R_T, R_lidar)
            # geometies.append(line_pcd)
            vis.add_geometry(line_pcd)
            vis.poll_events()
            vis.update_renderer()
            cv2.waitKey(10)
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
        meshfiles = os.listdir(plydir)
        mesh = TriangleMesh()
        imagedir = os.path.join(plydir, 'image')
        os.makedirs(imagedir, exist_ok=True)

        # rt_file = np.loadtxt(lidar_file, dtype=float)        
        # rotations = R.from_quat(rt_file[:, 4: 8]).as_matrix()  #3*3
        # translations = rt_file[:,1:4]
        _init_ = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        camera_init = np.array([[0.927565, -0.36788, 0.065483, -1.18345],
                        [-0.0171979, -0.217091, -0.976, -0.0448631],
                        [0.373267, 0.904177, -0.207693, 4.36933],
                        [0, 0, 0, 1]])
                        
        # camera_init = np.array([[1, 0, 0, -1.18345],
        #                         [0, 0, -1, -0.0448631],
        #                         [0, 1, 0, 4.36933],
        #                         [0, 0, 0, 1]])
        # camera_init = np.matmul(camera_init, _init_)
        for i in range(len(meshfiles) - 1):
            name = str(i)+'_smpl.ply'
            print('name', name)
            plyfile = os.path.join(plydir, name)

            with open(plyfile) as ply:
                plylines = ply.readlines()
            vertex_number = int(plylines[2].split(' ')[2])
            face_number = int(plylines[6].split(' ')[2])

            vertices = np.zeros((vertex_number, 3))
            faces = np.zeros((face_number, 3), dtype=int)

            for j in range(9, 9+vertex_number):
                vertices[j - 9] = np.asarray(plylines[j].strip().split(' '), dtype=float) + np.array([0, 0, 0.25])

            mesh.vertices = Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            
            # set_camera_pose
            # r = np.matmul(rotations[i], np.linalg.inv(rotations[0]))
            # r = np.matmul(camera_init, r)
            camera_pose = camera_init
            # camera_pose = toRt(rotations[i], translations[i])
            # camera_pose = np.linalg.inv(camera_pose)
            if i == 0:
                for f in range(9+vertex_number, 9+vertex_number + face_number):
                    faces[f - 9 - vertex_number] = np.asarray(plylines[f].strip().split(' '), dtype=int)[1:]
                mesh.triangles = Vector3iVector(faces)
                mesh.paint_uniform_color([75/255, 145/255, 183/255])
                vis.add_geometry(mesh)
                o3dcallback(camera_pose)
                vis.poll_events()
                vis.update_renderer()    
                cv2.waitKey(10)

            else:
                with Timer('update renderer'):
                    vis.update_geometry(mesh)
                    # t = translations[i] - translations[i-1]
                    # camera_pose[:3, 3] -= t 
                    # np.array([-t[2], -t[0], -t[1]])
                    # o3dcallback(camera_pose)
                    vis.poll_events()
                    vis.update_renderer()
                    cv2.waitKey(10)
                    outname = os.path.join(imagedir, '{:04d}.jpg'.format(i))
                    vis.capture_screen_image(outname)

    while True:
        with Timer('update renderer', True):
            # o3dcallback()
            vis.poll_events()
            vis.update_renderer()
            cv2.waitKey(10)
    # o3d.visualization.draw_geometries(geometry_list  = geometies, window_name = 'Draw RT')
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
 
