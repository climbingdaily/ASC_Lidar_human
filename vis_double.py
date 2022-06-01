import numpy as np
import h5py
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from o3dvis import o3dvis
import matplotlib.pyplot as plt
# from tool_func import imges_to_video
import torch
from smpl.smpl import SMPL
from vis_3d_box import load_data_remote

view = {
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 68.419929504394531, 39.271018981933594, 11.569537162780762 ],
			"boundingbox_min" : [ -11.513210296630859, -35.915927886962891, -2.4593989849090576 ],
			"field_of_view" : 60.0,
			"front" : [ 0.28886465410454343, -0.85891896928352873, 0.42286571841896009 ],
			"lookat" : [ 0.76326815774101275, 3.2896492351216851, 0.040108816664781548 ],
			"up" : [ -0.12866047345544837, 0.40286011796513765, 0.90617338734004726 ],
			"zoom" : 0.039999999999999994
		}
	],
}

pt_color = plt.get_cmap("tab20")(1)[:3]
smpl_color = plt.get_cmap("tab20")(3)[:3]
gt_smpl_color = plt.get_cmap("tab20")(5)[:3]

def poses_to_vertices(poses, trans=None, batch_size = 1024):
    poses = poses.astype(np.float32)
    vertices = np.zeros((0, 6890, 3))

    n = len(poses)
    smpl = SMPL()
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


def load_pkl_vis(file_path, start=0, end=-1, remote=False):
    """
    It loads the pickle file, converts the poses to vertices, and then visualizes the vertices and point
    clouds
    
    :param file_path: the path to the pickle file
    :param start: the first frame to visualize, defaults to 0 (optional)
    :param end: the last frame to be visualized
    :param remote: whether to load the data from a remote server, defaults to False (optional)
    """
    import pickle
    print(f'Load pkl in {file_path}')
    load_data_class = load_data_remote(remote)
    humans = load_data_class.load_pkl(file_path)

    first_person = humans['first_person']
    pose = first_person['pose']
    trans = first_person['trans']
    f_vert = poses_to_vertices(pose, trans)

    second_person = humans['second_person']    
    pose = second_person['pose']
    trans = second_person['trans']
    s_vert = poses_to_vertices(pose, trans)

    point_clouds = second_person['point_clouds']
    second_person['point_frame']
    ll = second_person['point_frame']
    point_valid_idx = [np.where(humans['frame_num'] == l)[0][0] for l in ll ]
    return f_vert, s_vert, point_clouds, point_valid_idx

    
def vis_pt_and_smpl(smpl_a, smpl_b, pc, pc_idx, vis):
    """
    > This function takes in two SMPL meshes, a point cloud, and a list of indices that correspond to
    the point cloud. It then displays the point cloud and the two SMPL meshes in a 3D viewer
    
    :param smpl_a: the SMPL mesh that you want to visualize
    :param smpl_b: the ground truth SMPL mesh
    :param pc: the point cloud data
    :param pc_idx: the index of the point cloud that you want to visualize
    """
    assert smpl_a.shape[0] == smpl_b.shape[0], "Groundtruth Data Shape are not compatible"
    pointcloud = o3d.geometry.PointCloud()
    p1 = o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')
    p2 = o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')  # a ramdon SMPL mesh

    init_param = False

    for i in range(smpl_a.shape[0]):

        # load data
        if i in pc_idx:
            pointcloud.points = o3d.utility.Vector3dVector(pc[pc_idx.index(i)])
        else:
            pointcloud.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
        p1.vertices = o3d.utility.Vector3dVector(smpl_a[i])
        p2.vertices = o3d.utility.Vector3dVector(smpl_b[i])
        
        # color
        pointcloud.paint_uniform_color(pt_color)
        p1.paint_uniform_color(smpl_color)
        p2.paint_uniform_color(gt_smpl_color)

        p1.compute_vertex_normals()
        p2.compute_vertex_normals()

        # add to visualization
        if not init_param:
            vis.change_pause_status()
            vis.add_geometry(pointcloud, reset_bounding_box = False)    
            vis.add_geometry(p1, reset_bounding_box = False)  
            vis.add_geometry(p2, reset_bounding_box = False)    
            init_param = True

        else:
            vis.vis.update_geometry(pointcloud) 
            vis.vis.update_geometry(p2)    
            vis.vis.update_geometry(p1)  

        vis.waitKey(40, helps=False)
        
    # vis.save_imgs(os.path.join(file_path, f'imgs'))
            
    # imges_to_video(os.path.join(file_path, f'imgs'), delete=True)

def load_scene(vis, pcd_path):
    from time import time
    reading_class = load_data_remote(remote=True)
    t1 = time()
    print(f'Loading scene from {pcd_path}')
    scene = reading_class.load_point_cloud(pcd_path)
    t2 = time()
    vis.set_view(view)
    print(f'====> Scene loading comsumed {t2-t1:.1f} s.')
    vis.add_geometry(scene)

if __name__ == '__main__':    
    parser = configargparse.ArgumentParser()
    parser.add_argument("--type", '-T', type=int, default=3)
    parser.add_argument("--start", '-S', type=int, default=0)
    parser.add_argument("--end", '-e', type=int, default=-1)
    parser.add_argument("--scene", '-s', type=str,
                        default='/hdd/dyd/lidarhumanscene/Scenes/0417-03_5cm.pcd')
    parser.add_argument("--remote", '-r', action='store_true')
    parser.add_argument("--file_path", '-F', type=str,
                        # default='C:\\Users\\DAI\\Desktop\\temp\\data\\pred_5.h5py')
                        default='/hdd/dyd/lidarhumanscene/0417-03/synced_data/two_person_param.pkl')
                        
    args, opts = parser.parse_known_args()

    vis = o3dvis(width=1280, height=720)

    load_scene(vis, args.scene)

    smpl_a, smpl_b, pc, pc_idx = load_pkl_vis(
        args.file_path, args.start, args.end, remote=True)

    vis_pt_and_smpl(smpl_a, smpl_b, pc, pc_idx, vis)
