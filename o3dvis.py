import open3d as o3d
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R
import os

colors = {
    'yellow':[251/255, 217/255, 2/255],
    'red'   :[234/255, 101/255, 144/255],
    'blue' :[27/255, 158/255, 227/255],
    'purple':[61/255, 79/255, 222/255],
    'blue2' :[75/255, 145/255, 183/255],
}

class Keyword():
    PAUSE = False       # pause the visualization
    DESTROY = False     # destory window
    REMOVE = False      # remove all geometies
    READ = False        # read the ply files
    VIS_TRAJ = False    # visualize the trajectory
    SAVE_IMG = False    # save the images in open3d window
    SET_VIEW = False    # set the view based on the info   
    VIS_STREAM = True   # only visualize the the latest mesh stream
    ROTATE = False      # rotate the view automatically




def o3d_callback_rotate():
    Keyword.ROTATE = not Keyword.ROTATE
    return False

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
    if ROTATE:
        camera['phi'] += np.pi/10
        camera_pose = set_camera(get_camera())
        # camera_pose = np.array([[-0.927565, 0.36788, 0.065483, -1.18345],
        #                         [0.0171979, 0.217091, -0.976, -0.0448631],
        #                         [-0.373267, -0.904177, -0.207693, 8.36933],
        #                         [0, 0, 0, 1]])
    print(camera_pose)
    init_camera(camera_pose)

def set_view(vis):
    Keyword.SET_VIEW = not Keyword.SET_VIEW
    print('SET_VIEW', Keyword.SET_VIEW)
    return False

def save_imgs(vis):
    Keyword.SAVE_IMG = not Keyword.SAVE_IMG
    print('SAVE_IMG', Keyword.SAVE_IMG)
    return False

def stream_callback(vis):
    # 以视频流方式，更新式显示mesh
    Keyword.VIS_STREAM = not Keyword.VIS_STREAM
    # print('VIS_STREAM', VIS_STREAM)
    return False

def pause_callback(vis):
    Keyword.PAUSE = not Keyword.PAUSE
    print('Pause', Keyword.PAUSE)
    return False

def destroy_callback(vis):
    Keyword.DESTROY = not Keyword.DESTROY
    return False

def remove_scene_geometry(vis):
    Keyword.REMOVE = not Keyword.REMOVE
    return False

def read_dir_ply(vis):
    Keyword.READ = not Keyword.READ
    print('READ', Keyword.READ)
    return False

def read_dir_traj(vis):
    Keyword.VIS_TRAJ = not Keyword.VIS_TRAJ
    print('VIS_TRAJ', Keyword.VIS_TRAJ)
    return False

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

def print_help(is_print=True):
    if is_print:
        print('============Help info============')
        print('Press SPACE to refresh visulization')
        print('Press Q to quit window')
        print('Press D to remove the scene')
        print('Press T to load and show traj file')
        print('Press F to stop current motion')
        print('Press . to turn on auto-screenshot ')
        print('Press , to set view zoom based on json file ')
        print('=================================')


class o3dvis():
    def __init__(self, window_name = 'DAI_VIS'):
        self.init_vis(window_name)
        print_help()

    def add_scene_gemony(self, geometry):
        if not Keyword.REMOVE:
            self.add_geometry(geometry)

    def init_vis(self, window_name):
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # vis.register_key_callback(ord(' '), pause_callback)
        self.vis.register_key_callback(ord("Q"), destroy_callback)
        self.vis.register_key_callback(ord("D"), remove_scene_geometry)
        self.vis.register_key_callback(ord(" "), read_dir_ply)
        self.vis.register_key_callback(ord("T"), read_dir_traj)
        self.vis.register_key_callback(ord("F"), stream_callback)
        self.vis.register_key_callback(ord("."), save_imgs)
        self.vis.register_key_callback(ord(","), set_view)
        self.vis.create_window(window_name=window_name, width=1280, height=720)

    def add_geometry(self, geometry, reset_bounding_box = True):
        self.vis.add_geometry(geometry, reset_bounding_box)

    def remove_geometry(self, geometry, reset_bounding_box = True):
        self.vis.remove_geometry(geometry, reset_bounding_box)

    def vis_wait_key(self, key, helps = True):
        print_help(helps)
        self.vis.poll_events()
        self.vis.update_renderer()
        cv2.waitKey(key)
        if Keyword.DESTROY:
            self.vis.destroy_window()
        return Keyword.READ

    def set_view_zoom(self, info, count, steps):
        """根据参数设置vis的视场角

        Args:
            vis ([o3d.visualization.VisualizerWithKeyCallback()]): [description]
            info ([type]): [description]
            count ([int]): [description]
            steps ([int]): [description]
        """        
        ctr = self.vis.get_view_control()
        elements = ['zoom', 'lookat', 'up', 'front', 'field_of_view']
        if 'step1' in info.keys():
            steps = info['step1']
        if 'views' in info.keys() and 'steps' in info.keys():
            views = info['views']
            fit_steps = info['steps']
            count += info['start']
            for i, v in enumerate(views):
                if i == len(views) - 1:
                    continue
                if count >= fit_steps[i+1]:
                    continue
                for e in elements:
                    z1 = np.array(views[i]['trajectory'][0][e])
                    z2 = np.array(views[i+1]['trajectory'][0][e])
                    if e == 'zoom':
                        ctr.set_zoom(z1 +(count - fit_steps[i])  * (z2-z1) / (fit_steps[i+1] - fit_steps[i]))
                    elif e == 'lookat':
                        ctr.set_lookat(z1 + (count - fit_steps[i]) * (z2-z1) / (fit_steps[i+1] - fit_steps[i]))
                    elif e == 'up':
                        ctr.set_up(z1 + (count - fit_steps[i]) * (z2-z1) / (fit_steps[i+1] - fit_steps[i]))
                    elif e == 'front':
                        ctr.set_front(z1 + (count - fit_steps[i]) * (z2-z1) / (fit_steps[i+1] - fit_steps[i]))
                break    
                
        # for e in elements:
        #     if count > steps:
        #         break
        #     z1 = np.array(info['view1']['trajectory'][0][e])
        #     z2 = np.array(info['view2']['trajectory'][0][e])
        #     if e == 'zoom':
        #         ctr.set_zoom(z1 + count * (z2-z1) / (steps - 1))
        #     elif e == 'lookat':
        #         ctr.set_lookat(z1 + count * (z2-z1) / (steps - 1))
        #     elif e == 'up':
        #         ctr.set_up(z1 + count * (z2-z1) / (steps - 1))
        #     elif e == 'front':
        #         ctr.set_front(z1 + count * (z2-z1) / (steps - 1))
        #     # elif e == 'field_of_view':
            #     ctr.change_field_of_view(z1 + count * (z2-z1) / (steps - 1))
    
    def add_mesh_by_order(self, plydir, mesh_list, color, strs='render', order = True, start=None, end=None, info=None):
        """[summary]

        Args:
            plydir ([str]): [description]
            mesh_list ([list]): [description]
            color ([str]): [red, yellow, green, blue]
            strs (str, optional): [description]. Defaults to 'render'.
            order (bool, optional): [description]. Defaults to True.
            start ([int], optional): [description]. Defaults to None.
            end ([int], optional): [description]. Defaults to None.
            info ([type], optional): [description]. Defaults to None.
        Returns:
            [list]: [A list of geometries]
        """        
        save_dir = os.path.join(plydir, strs)
        
        if order:
            num = np.array([int(m.split('_')[0]) for m in mesh_list], dtype=np.int32)
            idxs = np.argsort(num)
        else:
            idxs = np.arange(len(mesh_list))
        pre_mesh = None
        
        geometies = []
        helps = True
        count = 0

        # trajs = np.loadtxt('G:\\Human_motion\\visualization\\trajs\\campus_lidar_filt_synced_offset.txt')
        # mocap_trajs = np.loadtxt('G:\\Human_motion\\visualization\\trajs\\mocap_trans_synced.txt')

        sphere_list = []
        for i in idxs:
            # set view zoom
            if info is not None and Keyword.SET_VIEW:
                self.set_view_zoom(info, count, end-start)
            if order and end > start:
                if num[i] < start or num[i] > end:
                    continue

            plyfile = os.path.join(plydir, mesh_list[i])
            # print(plyfile)
            mesh = o3d.io.read_triangle_mesh(plyfile)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(colors[color])
            # mesh.vertices = Vector3dVector(np.array(mesh.vertices) - trajs[num[i],1:4] + mocap_trajs[num[i],1:4])
            if Keyword.VIS_STREAM and pre_mesh is not None:
                self.remove_geometry(pre_mesh, reset_bounding_box = False)
                geometies.pop()
            Keyword.VIS_STREAM = True
            geometies.append(mesh)
            self.add_geometry(mesh, reset_bounding_box = False)
                
            # if count % 5 == 0:
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            #     sphere.vertices = Vector3dVector(np.asarray(sphere.vertices) + trajs[num[i],1:4])
            #     sphere.compute_vertex_normals()
            #     sphere.paint_uniform_color(color)
            #     sphere_list.append(sphere)
            #     self.add_geometry(sphere, reset_bounding_box = False)

            
            pre_mesh = mesh
            if not self.vis_wait_key(10, helps=helps):
                break
            helps = False
            if Keyword.SAVE_IMG:
                os.makedirs(save_dir, exist_ok=True)
                
                outname = os.path.join( save_dir, strs + '_{:04d}.jpg'.format(count))
                self.vis.capture_screen_image(outname)
            count += 1
        for s in sphere_list:
            self.remove_geometry(s, reset_bounding_box = False)

        return geometies