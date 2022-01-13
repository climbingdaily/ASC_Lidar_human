import open3d as o3d
import numpy as np
import cv2
import sys
import os
# sys.path.insert(0, './')
# sys.path.insert(1, '../')

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh

yellow = [251/255, 217/255, 2/255]
red = [234/255, 101/255, 144/255]
blue = [27/255, 158/255, 227/255]
purple = [61/255, 79/255, 222/255]
# blue = [75/255, 145/255, 183/255]

def set_view_zoom(vis, zoom):
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)

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
        print('Press space to refresh mesh')
        print('Press Q to quit window')
        print('Press D to remove the scene')
        print('Press T to load and show traj file')
        print('Press F to stop current motion')
        print('Press . to turn on auto-screenshot ')
        print('=================================')

PAUSE = False
DESTROY = False
REMOVE = False
READ = False
VIS_TRAJ = False
SAVE_IMG = False
SET_VIEW = False
VIS_STREAM = True

rotate = False
def o3d_callback_rotate():
    global rotate
    rotate = not rotate
    return False

def set_view(vis):
    global SET_VIEW
    SET_VIEW = not SET_VIEW
    print('SET_VIEW', SET_VIEW)
    return False

def save_imgs(vis):
    global SAVE_IMG
    SAVE_IMG = not SAVE_IMG
    print('SAVE_IMG', SAVE_IMG)
    return False

def stream_callback(vis):
    # 以视频流方式，更新式显示mesh
    global VIS_STREAM
    VIS_STREAM = not VIS_STREAM
    # print('VIS_STREAM', VIS_STREAM)
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

vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.register_key_callback(ord(' '), pause_callback)
vis.register_key_callback(ord("Q"), destroy_callback)
vis.register_key_callback(ord("D"), remove_scene_geometry)
vis.register_key_callback(ord(" "), read_dir_ply)
vis.register_key_callback(ord("T"), read_dir_traj)
vis.register_key_callback(ord("F"), stream_callback)
vis.register_key_callback(ord("."), save_imgs)
vis.register_key_callback(ord("V"), set_view_zoom)
vis.register_key_callback(ord('A'), o3d_callback_rotate)
vis.create_window(window_name='Mesh vis', width=1920, height=1080)
# vis.create_window(window_name='RT', width=1280, height=720)

def vis_wait_key(vis, key, helps = True):
    global DESTROY, READ
    print_help(helps)
    vis.poll_events()
    vis.update_renderer()
    cv2.waitKey(key)
    if DESTROY:
        vis.destroy_window()
    # if SAVE_IMG:
    #     outname = os.path.join(plydir, 'render_' + strs, strs + '_{:04d}.jpg'.format(i))
    #     vis.capture_screen_image(outname)
    if READ:
        return True
    else:
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
    # if rotate:
    # camera['phi'] += np.pi/10
    # camera_pose = set_camera(get_camera())
    # camera_pose = np.array([[-0.927565, 0.36788, 0.065483, -1.18345],
    #                         [0.0171979, 0.217091, -0.976, -0.0448631],
    #                         [-0.373267, -0.904177, -0.207693, 8.36933],
    #                         [0, 0, 0, 1]])
    print(camera_pose)
    init_camera(camera_pose)
