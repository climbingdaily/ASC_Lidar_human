# from Viewer.viewer.viewer import Viewer
# from curses import flash
# from distutils.command.build_scripts import first_line_re
import numpy as np
import pickle as pkl
import os
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from o3dvis import o3dvis
import matplotlib.pyplot as plt

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

def load_point_cloud(file_name, pointcloud = None):
    if pointcloud is None:
        pointcloud = o3d.geometry.PointCloud()
    if not os.path.exists(file_name):
        pass
    elif file_name.endswith('.txt'):
        pts = np.loadtxt(file_name)
        pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3])  
    elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_name)
        pointcloud.points = pcd.points
        # print(len(pcd.poits))
        pointcloud.paint_uniform_color([0.5, 0.5, 0.5])
    else:
        pass
    return pointcloud

def load_boxes(dets, data_root_path = None):
    vis = o3dvis()
    pointcloud = o3d.geometry.PointCloud()
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
                                                                  0, 0, 0])
    vis.add_geometry(axis_pcd, reset_bounding_box = False)

    poses = np.loadtxt(os.path.join(data_root_path, 'poses.txt')).reshape(-1, 3, 4)

    boxes_list = []
    first_frame = True
    mesh_list = []

    for idx, frame_info in enumerate(dets):
        if idx <400:
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
        
        pointcloud = load_point_cloud(os.path.join(
            data_root_path, 'human_semantic', frame_id+'.pcd'), pointcloud)

        # update point cloud
        if first_frame:
            vis.add_geometry(pointcloud)
            first_frame = False
        else:
            vis.vis.update_geometry(pointcloud)

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
            data_root_path, 'segment_by_tracking'), f'{idx:04d}')
        if os.path.exists(mesh_dir):
            mesh_list += vis.add_mesh_together(
                mesh_dir, os.listdir(mesh_dir), plt.get_cmap("tab20")(idx)[:3])

        if len(pointcloud.points) > 0 and len(boxes_lidar) >0:

            for i, box in enumerate(boxes_lidar):

                transform = transformation[:3, :3] @ R.from_rotvec(
                    box[6] * np.array([0, 0, 1])).as_matrix()
                center = transformation[:3, :3] @ box[:3] + transformation[:3, 3]
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
    # parser.add_argument("--start_idx", '-S', type=int, default=0)
    # parser.add_argument("--end_idx", '-E', type=int, default=-1)
    parser.add_argument("--box_file_path", '-F', type=str,
                        default="C:\\Users\\DAI\\Desktop\\temp\\0417-03_tracking.pkl")
    args = parser.parse_args()


    with open(args.box_file_path, 'rb') as f:
        dets = pkl.load(f)

    # load_boxes(dets, 'C:\\Users\\Yudi Dai\\Desktop\\segment\\velodyne')
    # load_boxes(dets, 'C:\\Users\\DAI\\Desktop\\temp\\velodyne')
    load_boxes(dets, 'C:\\Users\\DAI\\Desktop\\temp')
