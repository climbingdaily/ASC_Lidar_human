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

yellow = [251/255, 217/255, 2/255]
red = [234/255, 101/255, 144/255]
blue = [27/255, 158/255, 227/255]
purple = [61/255, 79/255, 222/255]
# blue = [75/255, 145/255, 183/255]

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def load_render_option(vis):
    vis.get_render_option().load_from_json(
        "../../test_data/renderoption.json")
    return False

def capture_depth(vis):
    depth = vis.capture_depth_float_buffer()
    plt.imshow(np.asarray(depth))
    plt.show()
    return False

def capture_image(vis):
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    plt.show()
    return False


PAUSE = False
DESTROY = False
REMOVE = False
READ = False
VIS_TRAJ = False
VIS_STREAM = False

def stream_callback(vis):
    global VIS_STREAM
    VIS_STREAM = not VIS_STREAM
    print('VIS_STREAM', VIS_STREAM)
    return False

def pause_callback(vis):
    global PAUSE
    PAUSE = not PAUSE
    print('Pause', PAUSE)
    return False

def destroy_callback(vis):
    global DESTROY
    DESTROY = not DESTROY
    return False

def remove_scene_geometry(vis):
    global REMOVE
    REMOVE = not REMOVE
    return False

def read_dir_ply(vis):
    global READ
    READ = not READ
    print('READ', READ)
    return False

def read_dir_traj(vis):
    global VIS_TRAJ
    VIS_TRAJ = not VIS_TRAJ
    print('VIS_TRAJ', VIS_TRAJ)
    return False

def add_mesh_by_order(vis, plydir, mesh_list, color, order = True):
    global VIS_STREAM
    # append = mesh_list[0].split('_')[1:]
    # tail = '_'
    # for a in append:
    #     tail += a + '_'
    # tail = tail[:-1]
    if order:
        num = np.array([int(m.split('_')[0]) for m in mesh_list], dtype=np.int32)
        idxs = np.argsort(num)
    else:
        idxs = np.arange(len(mesh_list))
    pre_mesh = None
    
    geometies = []
    for i in idxs:
        plyfile = os.path.join(plydir, mesh_list[i])

        mesh = o3d.io.read_triangle_mesh(plyfile)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        if VIS_STREAM and pre_mesh is not None:
            vis.remove_geometry(pre_mesh, reset_bounding_box = False)
            geometies.pop()
        geometies.append(mesh)
        vis.add_geometry(mesh, reset_bounding_box = False)
        pre_mesh = mesh
        vis.poll_events()
        vis.update_renderer()    
        cv2.waitKey(10)
    return geometies

# key_to_callback = {}
# key_to_callback[ord("K")] = change_background_to_black
# key_to_callback[ord("R")] = load_render_option
# key_to_callback[ord(",")] = capture_depth
# key_to_callback[ord(".")] = capture_image
# key_to_callback[ord("T")] = pause_vis
# o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

# vis = o3d.visualization.Visualizer()
vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.register_key_callback(ord(' '), pause_callback)
vis.register_key_callback(ord("D"), destroy_callback)
vis.register_key_callback(ord("C"), remove_scene_geometry)
vis.register_key_callback(ord(" "), read_dir_ply)
vis.register_key_callback(ord("T"), read_dir_traj)
vis.register_key_callback(ord("V"), stream_callback)
# vis.register_key_callback(ord('A'), o3d_callback_rotate)
# vis.create_window(window_name='RT', width=1920, height=1080)
vis.create_window(window_name='RT', width=1280, height=720)

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

def load_scene(pcd_path):
    print('Loading scene...')
    scene_name = 'climbinggym1101'
    # scene_name = 'lab_building'
    # scene_name = 'campus'
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
    pcd_dir = 'E:\\SCSC_DATA\\HumanMotion\\scenes'
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
            if len(sys.argv) >=4:
                plydir2 = sys.argv[3]
            # lidar_file = sys.argv[3]
        else:
            print('python visualize_RT.py [-l] [lidar_traj_path]')
            print('python visualize_RT.py [-b] [bvh_path]')
            print('python visualize_RT.py [-c] [csv_pos_path] [csv_rot_path]')
            exit()
    geometies = []
    start_lidar_idx = int(np.loadtxt(lidar_file, dtype=np.float64)[0,0]) 
    positions = np.loadtxt(lidar_file, dtype=np.float64)[:, 1:4]
    
    scene_pcd = load_scene(pcd_dir)
    if not REMOVE:
        vis.add_geometry(scene_pcd)
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
            vis.poll_events()
            vis.update_renderer()
            cv2.waitKey(10)
            if DESTROY:
                vis.destroy_window()
            if READ:
                meshfiles = os.listdir(plydir)
                for mesh in mesh_list:
                    vis.remove_geometry(mesh, reset_bounding_box = False)
                mesh_list.clear()
                mesh_l1 = []
                mesh_l2 = []
                mesh_l3 = []
                mesh_l4 = []
                for plyfile in meshfiles:
                    if plyfile.split('.')[-1] != 'ply':
                        continue
                    print('name', plyfile)
                    if plyfile.split('_')[-1] == 'opt.ply':
                        mesh_l1.append(plyfile)
                    elif plyfile.split('_')[-1] == 'mocap.ply':
                        mesh_l2.append(plyfile)
                    elif plyfile.split('_')[-1] == 'smpl.ply':
                        mesh_l3.append(plyfile)
                    else:
                        mesh_l4.append(plyfile)
                mesh_list += add_mesh_by_order(vis, plydir, mesh_l1, red)
                mesh_list += add_mesh_by_order(vis, plydir, mesh_l2, yellow)
                mesh_list += add_mesh_by_order(vis, plydir, mesh_l3, blue)
                mesh_list += add_mesh_by_order(vis, plydir, mesh_l4, blue, order=False)
                READ = False
            if VIS_TRAJ:
                
                for sphere in sphere_list:
                    vis.remove_geometry(sphere, reset_bounding_box = False)
                sphere_list.clear()
                traj_files = os.listdir(plydir)
                for trajfile in traj_files:
                    if trajfile.split('.')[-1] != 'txt':
                        continue
                    print('name', trajfile)
                    if trajfile.split('_')[-1] == 'offset.txt':
                        color = red
                    elif trajfile.split('_')[-1] == 'synced.txt':
                        color = yellow
                    else:
                        color = blue
                    trajfile = os.path.join(plydir, trajfile)
                    trajs = np.loadtxt(trajfile)[:,1:4]
                    traj_cloud = o3d.geometry.PointCloud()
                    # show as points
                    traj_cloud.points = Vector3dVector(trajs)
                    traj_cloud.paint_uniform_color(color)
                    sphere_list.append(traj_cloud)
                    # for t in range(1400, 2100, 1):
                    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                    #     sphere.vertices = Vector3dVector(np.asarray(sphere.vertices) + trajs[t])
                    #     sphere.compute_vertex_normals()
                    #     sphere.paint_uniform_color(color)
                    #     sphere_list.append(sphere)
                for sphere in sphere_list:
                    vis.add_geometry(sphere, reset_bounding_box = False)
                    vis.poll_events()
                    vis.update_renderer()    
                    cv2.waitKey(1)
                VIS_TRAJ = False