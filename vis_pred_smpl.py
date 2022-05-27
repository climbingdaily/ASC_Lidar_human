import numpy as np
import h5py
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from o3dvis import o3dvis
import matplotlib.pyplot as plt
# from tool_func import imges_to_video
from smpl.smpl import SMPL
import torch

view = {
        "trajectory":
	[
		{
			"boundingbox_max" : [ 5.4838047865093849, 0.2540373302066915, 0.029168829321861267 ],
			"boundingbox_min" : [ 4.330204142923237, -1.7214865333566496, -1.8612730344932515 ],
			"field_of_view" : 60.0,
			"front" : [ -0.99257767585799128, 0.091189825013992476, 0.080461004233516153 ],
			"lookat" : [ 4.7962325322631543, -0.62380935711682217, -1.1889325160567368 ],
			"up" : [ 0.086773297396398955, 0.067507214866913023, 0.99393821276770966 ],
			"zoom" : 0.7599999999999989
		}
	],
}

def make_cloud_in_vis_center(point_cloud):
    center = point_cloud.get_center()
    yaw = np.arctan2(center[1], center[0])

    # rot the points, make them on the X-axis
    rt = R.from_rotvec(np.array([0, 0, -yaw])).as_matrix()

    # put points in 5 meters distance
    trans_x = 5 - (rt @ center)[0]

    rt = np.concatenate((rt.T, np.array([[trans_x, 0, 0]]))).T
    rt = np.concatenate((rt, np.array([[0, 0, 0, 1]])))

    point_cloud.transform(rt)
    # point_cloud.traslate(rt)

    return rt, center

def poses_to_vertices(poses, trans=None):
    poses = poses.astype(np.float32)
    vertices = np.zeros((0, 6890, 3))

    n = len(poses)
    smpl = SMPL()
    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]), torch.zeros((cur_n, 10)))
        vertices = np.concatenate((vertices, cur_vertices.cpu().numpy()))

    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)
    return vertices


def load_pkl_vis(file_path, start=0, end=-1):
    import pickle

    point_clouds = np.zeros((0, 512, 3))
    pred_vertices = np.zeros((0, 6890, 3))
    with open(file_path, "rb") as f:
        humans = pickle.load(f)
        for k,v in humans.items():
            
            pred_pose = v['pred_rotmats']
            if end == -1:
                end = pred_pose.shape[0]
            pred_pose = pred_pose[start:end]
            point_clouds = np.concatenate((point_clouds, v['point_clouds'][start:end]))
            pred_vertices = np.concatenate((pred_vertices, poses_to_vertices(pred_pose)))

    vis_pt_and_smpl(pred_vertices, point_clouds)



def load_hdf5_vis_it(file_path, start=0, end=-1):
    """
    载入h5py, 读取点云和pose, 以及trans
    It loads the vertices and point clouds from the hdf5 file
    
    :param [file_path]: the path to the hdf5 file
    :param [id_list]: a list of indices of the data you want to load
    :return: vertices and point_clouds
    """
    with h5py.File(file_path, mode='r') as f:
        # 'full_joints', 'lidar_to_mocap_RT', 'point_clouds', 'points_num', 'pose', 'rotmats', 'shape', 'trans'
        print(f.keys())
        pred_pose = f['pred_rotmats']
        if end == -1:
            end = pred_pose.shape[0]
        pred_pose = pred_pose[start:end]
        pose = f['gt_pose'][start:end]
        # gt_rotmats = f['gt_rotmats'][:]
        point_clouds = f['point_clouds'][start:end]

        vertices = poses_to_vertices(pose)
        # gt_vertices = poses_to_vertices(gt_rotmats)
        pred_vertices = poses_to_vertices(pred_pose)

        vis_pt_and_smpl(pred_vertices, point_clouds, vertices)
    
def vis_pt_and_smpl(pv, pc, v= None):
    # assert v.shape[0] == pc.shape[0], "Groundtruth Data Shape are not compatible"
    
    vis = o3dvis(width=600, height=600)
    pointcloud = o3d.geometry.PointCloud()
    gt = o3d.io.read_triangle_mesh(
        'C:\\Users\\DAI\\Documents\\GitHub\\ASC_Lidar_human\\smpl\\sample.ply')  # a ramdon SMPL mesh
    # gt_2 = o3d.io.read_triangle_mesh(
    #     'C:\\Users\\DAI\\Documents\\GitHub\\ASC_Lidar_human\\smpl\\sample.ply')  # a ramdon SMPL mesh
    pred = o3d.io.read_triangle_mesh(
        'C:\\Users\\DAI\\Documents\\GitHub\\ASC_Lidar_human\\smpl\\sample.ply')

    init_param = False

    for i in range(pv.shape[0]):

        # load data
        pointcloud.points = o3d.utility.Vector3dVector(pc[i])
        if v is not None:
            gt.vertices = o3d.utility.Vector3dVector(v[i])
        # gt_2.vertices = o3d.utility.Vector3dVector(gv[i])
        pred.vertices = o3d.utility.Vector3dVector(pv[i])
        
        # color
        pointcloud.paint_uniform_color(plt.get_cmap("tab20")(2)[:3])
        if v is not None:
            gt.paint_uniform_color(plt.get_cmap("tab20")(10)[:3])
        # gt_2.paint_uniform_color(plt.get_cmap("tab20")(15)[:3])
        pred.paint_uniform_color(plt.get_cmap("tab20")(15)[:3])


        # transform
        rt, center = make_cloud_in_vis_center(pointcloud) # 根据点的中心点，在XY平面将点云旋转平移
        rt[:3, 3] = np.array([5, -1, center[-1]])
        if v is not None:
            gt.transform(rt)
            gt.compute_vertex_normals()
        # rt[1, 3] = -2
        # gt_2.transform(rt)
        # rt[1, 3] = 1
        pred.transform(rt)

        # normals
        # gt_2.compute_vertex_normals()
        pred.compute_vertex_normals()

        # add to visualization
        if not init_param:
            vis.change_pause_status()
            vis.add_geometry(pointcloud, reset_bounding_box = True)    
            if v is not None:
                vis.add_geometry(gt)    
            # vis.add_geometry(gt_2)    
            vis.add_geometry(pred)  
            vis.set_view(view)
            init_param = True

        else:
            vis.vis.update_geometry(pointcloud) 
            if v is not None:
                vis.vis.update_geometry(gt)    
            # vis.vis.update_geometry(gt_2)    
            vis.vis.update_geometry(pred)  

        vis.waitKey(40, helps=False)
        
    # vis.save_imgs(os.path.join(file_path, f'imgs'))
            
    # imges_to_video(os.path.join(file_path, f'imgs'), delete=True)

if __name__ == '__main__':    
    parser = configargparse.ArgumentParser()
    parser.add_argument("--type", '-T', type=int, default=3)
    parser.add_argument("--start", '-S', type=int, default=0)
    parser.add_argument("--end", '-E', type=int, default=-1)
    parser.add_argument("--file_path", '-F', type=str,
                        # default='C:\\Users\\DAI\\Desktop\\temp\\data\\pred_5.h5py')
                        default='C:\\Users\\DAI\\Desktop\\temp\\0417-03\\segments.pkl')
                        
    args, opts = parser.parse_known_args()

    if args.file_path.endswith('.pkl'):
        load_pkl_vis(args.file_path, args.start, args.end)
    else:
        load_hdf5_vis_it(args.file_path, args.start, args.end)


