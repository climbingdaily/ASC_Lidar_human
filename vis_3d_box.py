# from Viewer.viewer.viewer import Viewer
# from curses import flash
# from distutils.command.build_scripts import first_line_re
# from tkinter import CENTER
# from typing_extensions import Self
from tkinter.messagebox import NO
import numpy as np
import pickle as pkl
import os
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# from sympy import re
from o3dvis import o3dvis
import matplotlib.pyplot as plt
from o3dvis import read_pcd_from_server, client_server, list_dir_remote

cmap = np.array([
    [245, 150, 100, 255],
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])
cmap = cmap[:, [2, 1, 0]]  # convert bgra to rgba

class load_data_remote(object):
    def __init__(self, remote):
        self.remote = remote
        if remote:
            self.client = client_server()
            self.sftp_client = self.client.open_sftp()

    
    def list_dir(self, folder):
        if self.remote:
            stdin, stdout, stderr = self.client.exec_command('ls ' + folder)
            res_list = stdout.readlines()
            dirs = [i.strip() for i in res_list]
        else:
            dirs = os.listdir(folder)
        return dirs

    def load_point_cloud(self, file_name, pointcloud = None, position = [0, 0, 0]):
        if pointcloud is None:
            pointcloud = o3d.geometry.PointCloud()
            
        if self.remote:
            # client = client_server()
            # files = sorted(list_dir_remote(client, file_path))
            _, stdout, _ = self.client.exec_command(f'[ -f {file_name} ] && echo OK') # 远程判断文件是否存在
            if stdout.read().strip() != b'OK':
                return pointcloud
        elif not os.path.exists(file_name):
            return pointcloud

        if file_name.endswith('.txt'):
            pts = np.loadtxt(file_name)
            pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3]) 
        elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
            if self.remote:
                pcd = read_pcd_from_server(self.client, file_name, self.sftp_client)
                pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
                if pcd.shape[1] == 6:
                    pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:]) 
            else:
                pcd = o3d.io.read_point_cloud(file_name)
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                rule1 = abs(points[:, 0] - position[0]) < 40
                rule2 = abs(points[:, 1] - position[1]) < 40
                rule3 = abs(points[:, 2] - position[2]) < 5
                rule = [a and b and c for a,b,c in zip(rule1, rule2, rule3)]
                
                pointcloud.points = o3d.utility.Vector3dVector(points[rule])
                pointcloud.colors = o3d.utility.Vector3dVector(colors[rule])

                # print(len(pcd.poits))
                # pointcloud.paint_uniform_color([0.5, 0.5, 0.5])
            # segment_ransac(pointcloud, return_seg=True)
            
        else:
            pass
        return pointcloud

    def load_pkl(self, filepath):
        if self.remote:
            # client = client_server()
            # sftp_client = client.open_sftp()

            with self.sftp_client.open(filepath, mode='rb') as f:
                dets = pkl.load(f)

        else:
            with open(filepath, 'rb') as f:
                dets = pkl.load(f)

        return dets

    def read_poses(self, data_root_path):
        
        if self.remote:
            
            # client = client_server()
            # sftp_client = client.open_sftp()
            with self.sftp_client.open(data_root_path + '/poses.txt', mode='r') as f:
                poses = f.readlines()

            ps = []
            for p in poses:
                ps.append(p.strip().split(' '))
            poses = np.asarray(ps).astype(np.float32).reshape(-1, 3, 4)
        else:
            poses = np.loadtxt(os.path.join(data_root_path, 'poses.txt')).reshape(-1, 3, 4)
        
        return poses

def segment_ransac(pointcloud, return_seg = False):
    # pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=20))
    pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    colors = np.asarray(pointcloud.colors)
    points = np.asarray(pointcloud.points)
    # normals = np.asarray(pointcloud.normals)


    rest_idx = set(np.arange(len(pointcloud.points)))
    # plane_idx = set()
    temp_idx = set()
    
    temp_cloud = o3d.geometry.PointCloud()

    for i in range(4):
        if i == 0:
            plane_model, inliers = pointcloud.segment_plane(distance_threshold=0.10, ransac_n=3, num_iterations=1200)
        elif len(temp_cloud.points) > 300:
            plane_model, inliers = temp_cloud.segment_plane(distance_threshold=0.10, ransac_n=3, num_iterations=1200)
        else:
            break
            
        # plane_inds += rest_idx[inliers]
        origin_inline = np.array(list(rest_idx))[inliers]
        colors[origin_inline] = plt.get_cmap('tab10')(i)[:3]
        rest_idx -= set(origin_inline)
        # plane_idx.union(set(origin_inline))

        if i == 0:
            temp_cloud = pointcloud.select_by_index(inliers, invert=True)
        else:
            temp_cloud = temp_cloud.select_by_index(inliers, invert=True)

        equation = plane_model[:3] ** 2
        if equation[2]/(equation[0] + equation[1]) < 130.6460956439: 
            # 如果平面与地面的夹角大于5°
            colors[origin_inline] = [1, 0, 0]
            temp_idx.union(set(origin_inline))

    if return_seg:
        non_ground_idx = np.array(list(rest_idx.union(temp_idx)))
        pointcloud.points = o3d.utility.Vector3dVector(points[non_ground_idx])
        pointcloud.colors = o3d.utility.Vector3dVector(colors[non_ground_idx])
        # pointcloud.normals = o3d.utility.Vector3dVector(normals[non_ground_idx])
    else:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

def load_boxes(dets, load_data, data_root_path = None, mesh_dir=None):
    vis = o3dvis()
    pointcloud = o3d.geometry.PointCloud()
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
                                                                  0, 0, 0])
    vis.add_geometry(axis_pcd, reset_bounding_box = False)

    poses = load_data.read_poses(data_root_path)

    boxes_list = []
    first_frame = True
    mesh_list = []

    join = '/' if load_data.remote else '\\'

    if mesh_dir is None:
        mesh_dir = join.join([data_root_path, 'segment_by_tracking_03_rot'])
    else:
        mesh_dir = join.join([data_root_path, mesh_dir])

    for idx, frame_info in enumerate(dets):
        if idx < 800:
            continue
        # transformation = poses[idx]
        
        if data_root_path is None:
            continue

        transformation = np.concatenate((poses[idx], np.array([[0,0,0,1]])))
        # transformation = np.eye(4)
        name = frame_info['name']
        score = frame_info['score']
        boxes_lidar = frame_info['boxes_lidar']
        frame_id = frame_info['frame_id']
        obj_id = frame_info['ids']
        # seq_id = frame_info['seq_id']

        name_m = name!='Car' #
        name = name[name_m]
        score = score[name_m]
        boxes_lidar = boxes_lidar[name_m][score>0.5]
        obj_id = obj_id[name_m][score>0.5]

        # print(boxes_lidar.shape)
        
        pointcloud = load_data.load_point_cloud(join.join(
            [data_root_path, 'human_semantic', frame_id+'.pcd']), pointcloud, position=transformation[:3, 3])

        # update axis
        vis.remove_geometry(axis_pcd, reset_bounding_box=False)
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
            0, 0, 0]).transform(transformation)
        vis.add_geometry(axis_pcd, reset_bounding_box=False)

        # remove pre boxes 
        for bbox in boxes_list:
            vis.remove_geometry(bbox, reset_bounding_box = False)
        boxes_list.clear()

        for mesh in mesh_list:
            vis.remove_geometry(mesh, reset_bounding_box = False)
        mesh_list.clear()

        # ! Temp code, for visualization test
        frame_mesh_dir = join.join([mesh_dir, f'{idx:04d}']) 
        if os.path.exists(frame_mesh_dir):
            mesh_list += vis.add_mesh_together(
                frame_mesh_dir, os.listdir(frame_mesh_dir), plt.get_cmap("tab20")(idx % 20)[:3])

        if len(pointcloud.points) > 0 and len(boxes_lidar) >0:
            for i, box in enumerate(boxes_lidar):

                # transform = transformation[:3, :3] @ R.from_rotvec(
                #     box[6] * np.array([0, 0, 1])).as_matrix()
                # center = transformation[:3, :3] @ box[:3] + transformation[:3, 3]
                transform = R.from_rotvec(
                    box[6] * np.array([0, 0, 1])).as_matrix()
                center = box[:3]
                extend = box[3:6]

                bbox = o3d.geometry.OrientedBoundingBox(center, transform, extend)

                bbox.color = cmap[int(obj_id[i]) % len(cmap)] / 255
                boxes_list.append(bbox)
                vis.add_geometry(bbox, reset_bounding_box = False, waitKey=0)
            
            # update point cloud
            if first_frame:
                vis.add_geometry(pointcloud)
                first_frame = False
                vis.change_pause_status()
            else:
                vis.vis.update_geometry(pointcloud)

            vis.waitKey(1, helps=False)
        else:
            print(f'Skip frame {idx}, {frame_id}')       

        vis.save_imgs(os.path.join(data_root_path, 'imgs'),
                        '{:04d}.jpg'.format(idx))

if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--remote", '-R', action='store_true')
    parser.add_argument("--tracking_file", '-B', type=str, default='C:\\Users\\DAI\\Desktop\\temp\\0417-03_tracking.pkl')
    parser.add_argument("--mesh_dir", '-M', type=str, default='New Folder')
    args = parser.parse_args() 

    # load_boxes(dets, 'C:\\Users\\Yudi Dai\\Desktop\\segment\\velodyne')
    # load_boxes(dets, 'C:\\Users\\DAI\\Desktop\\temp\\velodyne')
    load_data = load_data_remote(args.remote)

    dets = load_data.load_pkl(args.tracking_file)

    load_boxes(dets, load_data, os.path.dirname(args.tracking_file), mesh_dir = args.mesh_dir)