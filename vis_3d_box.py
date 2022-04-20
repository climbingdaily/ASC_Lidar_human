# from Viewer.viewer.viewer import Viewer
import numpy as np
import pickle as pkl
import os
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from o3dvis import o3dvis

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
        pointcloud.colors = pcd.colors
    else:
        pass
    return pointcloud

def load_boxes(dets, data_root_path = None):
    vis = o3dvis()
    pointcloud = o3d.geometry.PointCloud()
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
                                                                 0, 0, 0])

    poses = np.loadtxt(os.path.join(os.path.dirname(
        data_root_path), 'poses.txt')).reshape(-1, 3, 4)
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pointcloud, reset_bounding_box = False)
    boxes_list = []

    for idx, frame_info in enumerate(dets):
        # transformation = poses[idx]
        transformation = np.concatenate((poses[idx], np.array([[0,0,0,0]])))
        boxes_lidar = frame_info['boxes_lidar']
        print(boxes_lidar.shape)
        name = frame_info['name']
        frame_id = frame_info['frame_id']
        seq_id = frame_info['seq_id']
        score = frame_info['score']

        name_m = name!='Car' #
        name = name[name_m]
        boxes_lidar = boxes_lidar[name_m]
        score = score[name_m]

        boxes_lidar = boxes_lidar[score>0.6]

        if data_root_path is None:
            continue

        pointcloud = load_point_cloud(os.path.join(data_root_path, frame_id+'.pcd'), pointcloud)
        
        
        if len(pointcloud.points) > 0 and len(boxes_lidar) >0:
            for bbox in boxes_list:
                vis.remove_geometry(bbox, reset_bounding_box = False)
            boxes_list.clear()
            # axis = axis_pcd.transform(
            #     transformation)
            # vis.remove_geometry(axis_pcd, reset_bounding_box=False)
            # vis.add_geometry(axis, reset_bounding_box=False)
            vis.vis.update_geometry(pointcloud)
            for box in boxes_lidar:
                transform = R.from_rotvec(
                    box[6] * np.array([0, 0, 1])).as_matrix()
                bbox = o3d.geometry.OrientedBoundingBox(
                    box[:3, None], transform, box[3:6, None])
                # bbox = bbox.rotate(transformation[:3, :3]).translate(
                #     transformation[:3, 3])
                boxes_list.append(bbox)
            for bbox in boxes_list:
                vis.add_geometry(bbox, reset_bounding_box = False, waitKey=0)
            
            vis.waitKey(10, helps=False)
        
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
                        default="C:\\Users\\Yudi Dai\\Desktop\\segment\\result_1650273321.pkl")
    args = parser.parse_args()


    with open(args.box_file_path, 'rb') as f:
        dets = pkl.load(f)

    load_boxes(dets, 'C:\\Users\\Yudi Dai\\Desktop\\segment\\human_semantic')
