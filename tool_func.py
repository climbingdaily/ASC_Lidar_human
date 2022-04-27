import open3d as o3d
import numpy as np
import cv2
import sys
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
import random
import pickle as pkl
import argparse
import json
import paramiko
from subprocess import run

from visualization.Timer import Timer
from o3dvis import o3dvis, Keyword
from pypcd import pypcd

def cal_checkpoiont():
    # start point
    start = np.array([-0.012415, -0.021619, 0.912602])  # mocap
    b = np.array([-0.014763, -0.021510, 0.910694])  # mocap + lidar + filt
    c = np.array([-0.006008, -0.020295, 0.923018])  # slamcap

    d2 = np.linalg.norm(b - start)
    d3 = np.linalg.norm(c - start)
    print('start point in labbuilding')
    print('d2 ', d2)
    print('d3 ', d3)

    # end point. 12224
    a = np.array([-1.820513, -2.804140, 0.912960])  # mocap
    b = np.array([0.162808, -0.190505, 0.889144])  # mocap + lidar
    bb = np.array([0.139104, -0.109615, 0.861548])  # frame:12224  + filt
    c = np.array([0.147751, -0.105651, 0.921784])

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d3 = np.linalg.norm(c - start)
    print('end point in labbuilding')

    print('d1 ', d1)
    print('d2 ', d2)
    print('d3 ', d3)

    # frame 1612
    a = np.array([0.028645, -0.679929, 0.912945])  # mocap
    b = np.array([-0.024122, 0.028871, 0.975404])  # mocap + lidar
    bb = np.array([0.003263, -0.047339, 0.939227])  # filt
    c = np.array([0.040516, -0.068466, 0.917972])  # slamcap

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d22 = np.linalg.norm(bb - start)
    d3 = np.linalg.norm(c - start)
    print('CP1 point in labbuilding')

    print('d1 ', d1)
    print('d2 ', d2)
    print('d22 ', d22)
    print('d3 ', d3)

    # 9367
    start = np.array([57.381191, -25.599457, 1.035680])  # ground
    a = np.array([5.5611e+01, -2.4989e+01, -1.5531e-02])  # mocap
    b = np.array([57.2693, -25.4669,   0.9961])  # mocap+ ldiar
    bb = np.array([57.3397, -25.5093,   0.9585])  # mocap+ ldiar +filt
    c = np.array([57.3179, -25.5093,   1.0339])
    print('cp2 point in labbuilding')

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d22 = np.linalg.norm(bb - start)
    d3 = np.linalg.norm(c - start)
    print('d1 ', d1)
    print('d2 ', d2)
    print('d22 ', d22)
    print('d3 ', d3)

    # ============= campus ===========
    # start
    start = np.array([0.006690, 0.016684, 0.913541])  # mocap
    b = np.array([-0.030292, -0.130614, 0.940346])  # mocap + lidar
    bb = np.array([0.005844, 0.015523, 0.899268])  # mocap + lidar +filt
    c = np.array([0.003294, -0.059318, 0.937367])  # SLAMCap

    # 4599
    a = np.array([-1.819105, 0.616108, 0.913044])  # mocap
    b = np.array([0.068938, 0.905881, 0.925295])  # mocap + lidar
    bb = np.array([0.059497, 0.693883, 0.905785])  # mocap + lidar +filt
    c = np.array([0.029597, 0.912938, 0.930939])  # SLAMCap

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d22 = np.linalg.norm(bb - start)
    d3 = np.linalg.norm(c - start)
    print('end point in campus')

    print('d1 ', d1)
    print('d2 ', d2)
    print('d22 ', d22)
    print('d3 ', d3)


def cal_dist(file_path, idx):
    traj = np.loadtxt(file_path)
    st = traj[0, 0]
    idx = int(idx - st)
    dist = np.linalg.norm(traj[1:idx+1, 1:4] - traj[:idx, 1:4], axis=1).sum()
    print(f'{idx + st}: {dist:.3f}')

# _dir = 'E:\\SCSC_DATA\\HumanMotion\\HPS\\hps_txt'

# tot_dist = []
# txtfiles = os.listdir(_dir)
# count = 0
# for i, txt in enumerate(txtfiles):
#     if txt.split('_')[-1] == 'trans.txt':
#         count +=1
#         traj = np.loadtxt(os.path.join(_dir, txt))
#         dist = np.linalg.norm(traj[1:] - traj[:-1], axis=1).sum()
#         tot_dist.append(dist)
#         print(f'num {count} dist: {dist:.3f}')


def cal_mocap_smpl_trans():
    # slamcap, 1927 frams campus
    a = np.array([32.472198, 60.644207, -0.657228])
    # slamcap , 1832 frams campus
    aa = np.array([34.384434, 53.580666, -0.561240])
    # slamcap , 2949 frams campus
    aaa = np.array([-25.074957, 63.042736, 0.064208])

    # mocap, 1927 frames(10088) campus
    b = np.array([33.435452, 56.857124, 0.926840])
    # mocap, 1832 frames(9609) campus
    bb = np.array([34.662342, 50.486450, 0.900601])
    # mocap, 2949 frames(15162) campus
    bbb = np.array([-22.015713, 62.341846, 0.886846])
    print('1927: ', b-a)
    print('1832: ', bb-aa)
    print('2949: ', bbb-aaa)

# print('meand dist: ', np.asarray(tot_dist).mean())
# cal_checkpoiont()
# cal_mocap_smpl_trans()
# file_path = 'E:\\SCSC_DATA\\HumanMotion\\visualization\\lab_building_lidar_filt_synced_offset.txt'
# cal_dist(file_path, 1612)
# cal_dist(file_path, 12224)
# cal_dist(file_path, 9367)
# cal_dist(file_path, 9265)

def dbscan(self, file_path, file_name):
    pointcloud = o3d.geometry.PointCloud()
    if file_name.endswith('.txt'):
        pts = np.loadtxt(os.path.join(file_path, file_name))
        pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3])  
    elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(os.path.join(file_path, file_name))
        pointcloud.points = pcd.points
        pointcloud.colors = pcd.colors

    labels = np.array(pointcloud.cluster_dbscan(eps=0.1, min_points=20))
    max_label = labels.max()
    for i in range(max_label):
        list[np.where(labels == i)[0]]
    print(f"point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

def make_bounding_box():
    """
    # 测试open3d的boundingbox函数
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(
        "C:\\Users\\Daiyudi\\Desktop\\temp\\001121.pcd")
    people = o3d.io.read_point_cloud(
        "C:\\Users\\Daiyudi\\Desktop\\temp\\001121.ply")
    print(pcd)  # 输出点云点的个数
    print(people)  # 输出点云点的个数
    aabb = people.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # aabb包围盒为红色
    obb = people.get_oriented_bounding_box()
    obb.color = (0, 1, 0)  # obb包围盒为绿色

    seg_pcd = pcd.crop(obb)
    bg = pcd.select_by_index(
        obb.get_point_indices_within_bounding_box(pcd.points), invert=True)
    axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=aabb.get_center())
    axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=obb.get_center())
    axis2 = axis2.rotate(obb.R)
    people.paint_uniform_color([0, 0, 1])
    seg_pcd.paint_uniform_color([1, 0, 1])
    bg.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries(
        [bg, people, seg_pcd, axis1, axis2, aabb, obb])

def read_pcd_from_server(filepath):
    hostname = "10.24.80.240"
    port = 511
    username = 'dyd'
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, compress=True)
    sftp_client = client.open_sftp()
    # "/hdd/dyd/lidarcap/velodyne/crop_6/000001.pcd"
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        pc = np.zeros((pc_pcd.pc_data.shape[0], 4))
        pc[:, 0] = pc_pcd.pc_data['x']
        pc[:, 1] = pc_pcd.pc_data['y']
        pc[:, 2] = pc_pcd.pc_data['z']
        pc[:, 3] = pc_pcd.pc_data['intensity']
        return pc
    except Exception as e:
        print(f"Load {filepath} error")
    finally:
        remote_file.close()

def imges_to_video(path):
    """输入文件夹图片所在文件夹, 利用ffmpeg生成视频

    Args:
        path (str): [图片所在文件夹]
    """            
    strs = path.split('\\')[-1]
    # strs = sys.argv[2]
    video_path = os.path.join(os.path.dirname(path), strs + '.mp4')
    video_path2 = os.path.join(os.path.dirname(path), strs + '.avi')

    # command = f"ffmpeg -f image2 -i {path}\\{strs}_%4d.jpg -b:v 10M -c:v h264 -r 20  {video_path}"
    command = f"ffmpeg -f image2 -i {path}\\%4d.jpg -b:v 10M -c:v h264 -r 30  {video_path2}"
    if not os.path.exists(video_path) and not os.path.exists(video_path2):
        run(command, shell=True)

def read_pkl_file(file_name):
    """
    Reads a pickle file
    Args:
        file_name:
    Returns:
    """
    with open(file_name, 'rb') as f:
        data = pkl.load(f)
    return data

def save_hps_file_to_txt(hps_filepath):
    """ 把hps的文件转成txt, 方便可视化
    Args:
        hps_filepath (str): 包含hps的文件夹
    """    
    results_list = os.listdir(hps_filepath)
    for r in results_list:
        rf = os.path.join(hps_filepath, r)
        if r.split('.')[-1] == 'pkl':
            results = read_pkl_file(filepath)
            trans = results['transes']
            np.savetxt(filepath.split('.')[0]+'.txt', trans)
            print('save: ', filepath.split('.')[0]+'.txt')  

lidar_cap_view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : 'false',
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 14.874032020568848, 76.821174621582031, 3.0000000000000004 ],
			"boundingbox_min" : [ -15.89847469329834, -0.17999999999999999, -9.3068475723266602 ],
			"field_of_view" : 60.0,
			"front" : [ -0.31752652860060709, -0.88020048903111714, 0.3527378668986792 ],
			"lookat" : [ -4.6463824978204959, 2.0940846184404625, 0.24133203465013156 ],
			"up" : [ 0.077269992865328138, 0.34673394080716397, 0.9347753326307483 ],
			"zoom" : 0.059999999999999998
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

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

def ransac(scene_pts, distance_threshold=0.05):
    from util.segmentation import Segmentation
    seg_pcd = Segmentation(scene_pts)
    plane_equations, segments, segments_idx, rest = seg_pcd.run(1, distance_threshold)

def visulize_scene_with_meshes(plydir, pcd_dir, scene_name):
    """ 载入场景点云和生成的human meshes

    Args:
        plydir (str): [description]
        pcd_dir (str): [description]
        scene_name (str): [description]
    """    
    scene_pcd = load_scene(pcd_dir, scene_name)
    vis = o3dvis()
    vis.add_scene_gemony(scene_pcd)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-T', type=int, default = 3,
                        help='Function type')

    parser.add_argument('--hps-file', action='store_true',
                        help='run directory',)
    parser.add_argument('--remote-dir', action='store_true',
                        help='The remoted directory of pcd files')
    parser.add_argument('--velodyne-dir', action='store_true',
                        help='The local directory of pcd files')
    parser.add_argument('--img-dir', action='store_true',
                        help='A directroy containing imgs')
    parser.add_argument('--mesh-dir', action='store_true',
                        help='A directroy containing human mesh files')
    parser.add_argument('--scene-dir', action='store_true',
                        help='A directroy containing scene pcd files')
    parser.add_argument('--scene', action='store_true',
                        help="scene's file name")

    # 输入待读取的文件夹
    parser.add_argument('--file-name', '-f', type=str,
                        help='A file or a directory', default=None)
    args, opts = parser.parse_known_args()

    hps_file='E:\\SCSC_DATA\\HumanMotion\\HPS\\result'
    velodyne_dir='C:\\Users\\Yudi Dai\\Desktop\\segment\\pcds'
    scene_dir='J:\\Human_motion\\visualization'
    mesh_dir='J:\\Human_motion\\visualization\\climbinggym002_step_1'
    scene='climbinggym002'
    remote_dir='/hdd/dyd/lidarhumanscene/0417-03/human_semantic'

    # 读取hps的轨迹，保存成txt
    if args.hps_file:
        if args.file_name:
            hps_file = args.file_name
        save_hps_file_to_txt(hps_file)

    # 将图片保存成视频
    if args.img_dir:
        imges_to_video(args.file_name)
    
    # 可视化分割结果的点云
    if args.velodyne_dir:
        # pcd visualization
        if args.file_name:
            velodyne_dir = args.file_name
        vis = o3dvis()
        vis.visulize_point_clouds(velodyne_dir, skip=0, view=lidar_cap_view)

    # 可视化场景和human mesh
    if args.mesh_dir:
        if args.file_name:
            mesh_dir = args.file_name
        visulize_scene_with_meshes(mesh_dir, scene_dir, scene)

    # 可视化远程的文件夹
    if args.remote_dir:
        # visualize the pcd files on the remote server
        if args.file_name:
            remote_dir = args.file_name
        vis = o3dvis()
        vis.visulize_point_clouds(remote_dir, skip=0, view=lidar_cap_view, remote = True)

