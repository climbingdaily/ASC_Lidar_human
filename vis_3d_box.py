# from Viewer.viewer.viewer import Viewer
# from curses import flash
# from distutils.command.build_scripts import first_line_re
from tkinter import CENTER
import numpy as np
import pickle as pkl
import os
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sympy import re
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

def load_point_cloud(file_name, pointcloud = None, remote = False, position = [0, 0, 0]):
    if pointcloud is None:
        pointcloud = o3d.geometry.PointCloud()
        
    if remote:
        client = client_server()
        # files = sorted(list_dir_remote(client, file_path))
        _, stdout, _ = client.exec_command(f'[ -f {file_name} ] && echo OK') # 远程判断文件是否存在
        if stdout.read().strip() != b'OK':
            return pointcloud
    elif not os.path.exists(file_name):
        return pointcloud

    if file_name.endswith('.txt'):
        pts = np.loadtxt(file_name)
        pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3]) 
    elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
        if remote:
            pcd = read_pcd_from_server(client, file_name)
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
        segment_ransac(pointcloud, return_seg=True)
        
    else:
        pass
    return pointcloud

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

def load_boxes(dets, data_root_path = None, remote = False):
    vis = o3dvis()
    pointcloud = o3d.geometry.PointCloud()
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
                                                                  0, 0, 0])
    vis.add_geometry(axis_pcd, reset_bounding_box = False)

    if remote:
        
        client = client_server()
        sftp_client = client.open_sftp()
        with sftp_client.open(data_root_path + '/poses.txt', mode='r') as f:
            poses = f.readlines()

        ps = []
        for p in poses:
            ps.append(p.strip().split(' '))
        poses = np.asarray(ps).astype(np.float32).reshape(-1, 3, 4)
    else:
        poses = np.loadtxt(os.path.join(data_root_path, 'poses.txt')).reshape(-1, 3, 4)

    boxes_list = []
    first_frame = True
    mesh_list = []

    for idx, frame_info in enumerate(dets):
        # if idx <400:
        #     continue
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
        join = '/' if remote else '\\'
        
        pointcloud = load_point_cloud(join.join(
            [data_root_path, 'human_semantic', frame_id+'.pcd']), pointcloud, remote=remote, position=transformation[:3, 3])

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
        mesh_dir = os.path.join(os.path.join(
            data_root_path, 'segment_by_tracking_01'), f'{idx:04d}')
        if os.path.exists(mesh_dir):
            mesh_list += vis.add_mesh_together(
                mesh_dir, os.listdir(mesh_dir), plt.get_cmap("tab20")(idx % 20)[:3])

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

                # if score[score>0.6][i] >= 0.9:
                #     bbox.color = np.array([0, 1, 0])
                # elif score[score>0.6][i] >= 0.8:
                #     bbox.color = np.array([0, 0, 1])
                # elif score[score>0.6][i] >= 0.6:
                #     bbox.color = np.array([1, 0, 0])

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

            vis.waitKey(10, helps=False)
        else:
            print(f'Skip frame {idx}, {frame_id}')       

        vis.save_imgs(os.path.join(data_root_path, 'imgs'),
                        '{:04d}.jpg'.format(idx))
        
        # paths = os.path.join(data_root_path, str(seq_id), frame_id+'.npy')
        # if os.path.exists(paths):
            # points = np.load(paths)
            # vi.add_points(points[:,0:3])
            # vi.add_3D_boxes(boxes_lidar)
            # vi.show_3D()


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--remote", '-R', action='store_true')
    parser.add_argument("--box_dir", '-B', type=str, default='C:\\Users\\Yudi Dai\\Desktop\\segment')
    args = parser.parse_args() 


    # load_boxes(dets, 'C:\\Users\\Yudi Dai\\Desktop\\segment\\velodyne')
    # load_boxes(dets, 'C:\\Users\\DAI\\Desktop\\temp\\velodyne')
    if args.remote:
        client = client_server()
        sftp_client = client.open_sftp()

        with sftp_client.open(
            args.box_dir + f'/{os.path.basename(args.box_dir)}_tracking.pkl', mode='rb') as f:
            dets = pkl.load(f)

        load_boxes(dets, args.box_dir, remote=True)
    else:
        with open(os.path.join(args.box_dir, 'tracking_results.pkl'), 'rb') as f:
            dets = pkl.load(f)
        load_boxes(dets, args.box_dir)
