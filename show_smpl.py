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

# key_to_callback = {}
# key_to_callback[ord("K")] = change_background_to_black
# key_to_callback[ord("R")] = load_render_option
# key_to_callback[ord(",")] = capture_depth
# key_to_callback[ord(".")] = capture_image
# key_to_callback[ord("T")] = pause_vis
# o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

# vis = o3d.visualization.Visualizer()
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_key_callback(ord(' '), pause_callback)
vis.register_key_callback(ord("D"), destroy_callback)
vis.register_key_callback(ord("V"), remove_scene_geometry)
# vis.register_key_callback(ord('A'), o3d_callback_rotate)
# vis.create_window(window_name='RT', width=1920, height=1080)
vis.create_window(window_name='RT', width=1280, height=720)

def crop_scene(kdtree, scene_pcd, position):
    position[-1] -= 0.8
    [_, idx, _] = kdtree.search_radius_vector_3d(position, radius = 0.6)
    return scene_pcd.select_by_index(idx)

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

def load_scene(pcd_path, file_name):
    print('Loading scene...')
    scene_pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, file_name + '.pcd'))
    print('Loading normals...')
    
    with open(os.path.join(pcd_path, file_name + '_normals.pkl'), 'rb') as f:
        normals = pkl.load(f)
    scene_pcd.normals = o3d.utility.Vector3dVector(normals)
    # scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=80))
    # np.savetxt(os.path.join(pcd_path, 'scene_normals.txt'), np.asarray(scene_pcd.normals), fmt='%.6f')
    print('Building KDtreee...')
    kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
    scene_pcd.voxel_down_sample(voxel_size=0.02)
    print('Scene loaded...')
    return scene_pcd, kdtree

if __name__ == "__main__":

    show_grid = False
    file_name = 'lab_building'
    lidar_file = "E:\\SCSC_DATA\HumanMotion\\visualization\\" + file_name + "_lidar_filt_synced_offset.txt"
    plydir = 'E:\\SCSC_DATA\HumanMotion\\1023\\SMPL\\shiyanlou002_step_1'
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
            start_lidar_idx = int(np.loadtxt(lidar_file, dtype=np.float64)[0,0])
            positions = np.loadtxt(lidar_file, dtype=np.float64)[:, 1:4]
            scene_pcd, kdtree = load_scene(pcd_dir, file_name)
            if not REMOVE:
                vis.add_geometry(scene_pcd)
        else:
            print('python visualize_RT.py [-l] [lidar_traj_path]')
            print('python visualize_RT.py [-b] [bvh_path]')
            print('python visualize_RT.py [-c] [csv_pos_path] [csv_rot_path]')
            exit()
    geometies = []
    if key == '-l':
        rt_file = np.loadtxt(lidar_file, dtype=float)        
        R_init = R.from_quat(rt_file[0, 4: 8]).as_matrix()  #3*3
        for i in range(0, 500, 20):
            # 读取 i 帧的 RT
            R_lidar = R.from_quat(rt_file[i, 4: 8]).as_matrix()  #3*3
            # R_lidar = np.matmul(R_lidar, np.linalg.inv(R_init)) # 乘第一帧的逆
            R_T = rt_file[i, 1:4].reshape(1,3)   #1*3
            R_lidar = R_lidar.T + R_T
            line_pcd, point_pcd = triangle_pcd(R_T, R_lidar)
            # geometies.append(line_pcd)
            vis.add_geometry(line_pcd, reset_bounding_box=False)
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

            vis.add_geometry(line_pcd, reset_bounding_box=False)
            vis.poll_events()
            vis.update_renderer()
            cv2.waitKey(10)
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
        # R_init = mocap_init @ R_init

        for i in range(0, mocap_length, 100):
            R_mocap = R.from_euler('yxz', rot_data[i, 1:4], degrees=True).as_matrix()
            T_mocap = pos_data[i, 0].copy()
            
            # R_mocap = mocap_init @ R_mocap
            # T_mocap = mocap_init @ T_mocap

            R_mocap = R_mocap @ np.linalg.inv(R_init)
            R_mocap = R_mocap.T + T_mocap
            line_pcd, point_pcd = triangle_pcd(T_mocap, R_mocap)
            if i == 0:
                vis.add_geometry(line_pcd)
            else:
                vis.add_geometry(line_pcd, reset_bounding_box=False)

            vis.poll_events()
            vis.update_renderer()
            cv2.waitKey(10)
            geometies.append(line_pcd)
    elif key == '-m':
        meshfiles = os.listdir(plydir)
        ply_list = np.asarray([i.split('_')[0] for i in meshfiles], dtype=np.int64)
        ply_list.sort()
        # sort meshfiles

        if len(sys.argv) >= 4:
            meshfiles_compare = os.listdir(plydir2)
        mesh = TriangleMesh()
        mesh_compare = TriangleMesh()
        imagedir = plydir + '_render_images'
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

        with open('.\\vertices\\all_new.json') as f:
            all_vertices = json.load(f)
        back = all_vertices['back_new']
        left_heel = all_vertices['left_heel']
        left_toe = all_vertices['left_toe']
        right_heel = all_vertices['right_heel']
        right_toe = all_vertices['right_toe']

        grid = o3d.geometry.PointCloud()
        grid_list = []
        from util.segmentation import Segmentation
        initialized = False
        for i, idx in enumerate(ply_list):
            name = str(idx)+'_smpl.ply'
            print('name', name)
            plyfile = os.path.join(plydir, name)

            # =============================================
            # load smpl vertices
            # =============================================
            with open(plyfile) as ply:
                plylines = ply.readlines()
            vertex_number = int(plylines[2].split(' ')[2])
            face_number = int(plylines[6].split(' ')[2])

            vertices = np.zeros((vertex_number, 3))
            faces = np.zeros((face_number, 3), dtype=int)

            for j in range(9, 9+vertex_number):
                vertices[j - 9] = np.asarray(plylines[j].strip().split(' '), dtype=float) + np.array([0, 0, 0])

            mesh.vertices = Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            
            
            # =============================================
            # Compute grid for currunt pose
            # =============================================
            if show_grid:
                grid_file = os.path.join(plydir + '_grid', f'grid_{idx}.pcd')
                os.makedirs(os.path.join(plydir + '_grid'), exist_ok=True)
                if os.path.exists(grid_file):
                    grid = o3d.io.read_point_cloud(grid_file)
                else:
                    grid = crop_scene(kdtree, scene_pcd, positions[idx - start_lidar_idx])
                    o3d.io.write_point_cloud(grid_file, grid)
                grid.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

                # seg_grid = Segmentation(grid)   # Grid segmentation

                # _, segments, _, _ = seg_grid.run(10, 0.01)
                for g in grid_list:
                    vis.remove_geometry(g, reset_bounding_box=False)
                grid_list.clear()
                grid_list.append(grid)
                # for seg in segments:
                #     grid_list.append(seg)
                grid.paint_uniform_color([1,0,0])

            if len(sys.argv) >= 4:
                plyfile = os.path.join(plydir2, name)

                with open(plyfile) as ply:
                    plylines = ply.readlines()
                vertex_number = int(plylines[2].split(' ')[2])
                face_number = int(plylines[6].split(' ')[2])

                vertices = np.zeros((vertex_number, 3))
                faces_compare = np.zeros((face_number, 3), dtype=int)

                for j in range(9, 9+vertex_number):
                    vertices[j - 9] = np.asarray(plylines[j].strip().split(' '), dtype=float) + np.array([0, 0, 0])

                mesh_compare.vertices = Vector3dVector(vertices)
                mesh_compare.compute_vertex_normals()

            camera_pose = camera_init
            if not initialized:
                for f in range(9+vertex_number, 9+vertex_number + face_number):
                    faces[f - 9 - vertex_number] = np.asarray(plylines[f].strip().split(' '), dtype=int)[1:]
                mesh.triangles = Vector3iVector(faces)
                # verts_color = np.zeros((6890, 3)) + np.asarray([75/255, 145/255, 183/255])
                 # verts_color[back['verts']] = np.asarray([1 ,1, 1])
                # verts_color[left_heel['verts']] = np.asarray([1 ,1, 1])
                # verts_color[left_toe['verts']] = np.asarray([1 ,1, 1])
                # verts_color[right_heel['verts']] = np.asarray([1 ,1, 1])
                # verts_color[right_toe['verts']] = np.asarray([1 ,1, 1])
                # mesh.vertex_colors = Vector3dVector(verts_color)
                mesh.paint_uniform_color([75/255, 145/255, 183/255])
                box = mesh.get_axis_aligned_bounding_box()
                box.color = (0, 1, 0)
                vis.add_geometry(mesh)
                # vis.add_geometry(grid)
                for seg in grid_list:
                    vis.add_geometry(seg)

                if len(sys.argv) >= 4:
                    for f in range(9+vertex_number, 9+vertex_number + face_number):
                        faces_compare[f - 9 - vertex_number] = np.asarray(plylines[f].strip().split(' '), dtype=int)[1:]
                    mesh_compare.triangles = Vector3iVector(faces_compare)
                    mesh_compare.paint_uniform_color([239/255, 105/255, 102/255])
                    vis.add_geometry(mesh_compare)
                o3dcallback(camera_pose)
                vis.poll_events()
                vis.update_renderer()    
                cv2.waitKey(10)
                initialized = True

            else:
                with Timer('update renderer'):
                    vis.update_geometry(mesh)
                    
                    for seg in grid_list:
                        vis.add_geometry(seg, reset_bounding_box=False)
                    # vis.update_geometry(grid)
                    if len(sys.argv) >= 4:
                        vis.update_geometry(mesh_compare)
                    # t = translations[i] - translations[i-1]
                    # camera_pose[:3, 3] -= t 
                    # np.array([-t[2], -t[0], -t[1]])
                    # o3dcallback(camera_pose)
                    
                    vis.poll_events()
                    vis.update_renderer()
                    cv2.waitKey(10)

                    while PAUSE:
                        vis.poll_events()
                        vis.update_renderer()
                        cv2.waitKey(10)
                    if DESTROY:
                        vis.destroy_window()
                    if REMOVE:
                        vis.remove_geometry(scene_pcd, reset_bounding_box = False)
                        REMOVE = False
                    outname = os.path.join(imagedir, '{:04d}.jpg'.format(i))
                    vis.capture_screen_image(outname)
    
    while True:
        with Timer('update renderer', True):
            # o3dcallback()
            vis.poll_events()
            vis.update_renderer()
            cv2.waitKey(10)
            if DESTROY:
                vis.destroy_window()
