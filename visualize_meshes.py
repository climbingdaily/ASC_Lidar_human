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

if __name__ == "__main__":
    
    plydir = 'E:\\SCSC_DATA\HumanMotion\\visualization\\contact_compare\\\climbing'
    # pcd_dir = 'E:\\SCSC_DATA\\HumanMotion\\scenes'
    pcd_dir = 'J:\\Human_motion\\visualization'
    print('python visualize_RT.py [-c] [plydir] [scene_name]')
 
    key = sys.argv[1]
    plydir = sys.argv[2]
    scene_name = sys.argv[3]
    
    # scene_name = 'climbinggym1101'
    # scene_name = 'lab_building'
    # scene_name = 'campus'

    visulize_scene_with_meshes(plydir, pcd_dir, scene_name)
    