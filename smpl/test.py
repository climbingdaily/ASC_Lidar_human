import numpy as np
import os
import pytorch3d.loss
from model import SMPL
import sys
import torch
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_PATH)

from io3d import mocap, pcd
from util import transformation

transf = np.array([-0.571859717369, -0.820294380188, 0.009676335379, -1.879348874092,
                   0.817404925823, -0.570764899254, -0.077952407300, -3.341531515121,
                   0.069466829300, -0.036668356508, 0.996910095215, -0.333962023258,
                   0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]).reshape(4, 4)

# pc_rotated_numpy has been aligned into the mocap coordinate
# mocap_points generated from _rotations.csv
# pose_numpy generated from _worldpos.csv


def refine_shape(mocap_data: mocap.MoCapData, index: int, pc_numpy: np.array, num_iterations: int = 10) -> np.array:
    transf_torch = torch.from_numpy(transf)
    pc = torch.from_numpy(pc_numpy).float().cuda()
    beta = torch.zeros((1, 10))
    if pc.ndimension() == 3:
        assert pc.shape[0] == 1
    elif pc.ndimension() == 2:
        pc.unsqueeze_(0)
    else:
        print('ndimensions of pc is wrong!')
        exit(1)

    model = SMPL().cuda()
    model.requires_grad_(False)
    optimizer = torch.optim.Adam([beta.requires_grad_()], lr=2e-1)
    run = [0]
    while run[0] <= num_iterations:
        def closure():
            vertices = mocap_data.smpl_vertices(
                index, beta=beta, is_torch=True)
            vertices = transformation.affine(
                vertices, transf_torch).unsqueeze(0).float()
            loss1, _ = pytorch3d.loss.chamfer_distance(pc, vertices)
            loss2, _ = pytorch3d.loss.chamfer_distance(vertices, pc)
            loss = loss1 + loss2
            print('{}: {}'.format(run[0], loss))
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)
    return beta.detach().numpy()


# the input lidar_points must be cropped


def coarse_align(mocap_vertices, lidar_points):
    mocap_vertices = transformation.affine(mocap_vertices, transf)
    # mocap_vertices_cloud = pcl.PointCloud(mocap_vertices.astype(np.float32))
    # lidar_cloud = pcl.PointCloud(lidar_points)

    # icp = mocap_vertices_cloud.make_IterativeClosestPoint()
    # transf = icp.icp(mocap_vertices_cloud, lidar_cloud)[1]  # 4 * 4 ndarray
    # print(transf)
    # mocap_vertices = transformation.affine(mocap_vertices, transf)
    # from io3d import pcd

    pcd.save_point_cloud(
        '/home/ljl/ASC_Lidar_human/tmp/transformed.pcd', mocap_vertices)


if __name__ == '__main__':

    from util import pc_util
    lidar_points = pcd.read_point_cloud(
        '/home/ljl/ASC_Lidar_human/tmp/merged2.pcd')
    # lidar_points = pc_util.crop_points(
    #     lidar_points, {'min': [3.851, 1.375], 'max': [4.445, 2.077]})
    mocap_data = mocap.MoCapData('/xmu_gait/raw/1/shape/take002_chr00_worldpos.csv',
                                 '/xmu_gait/raw/1/shape/take002_chr00_rotations.csv')

    # pcd.save_point_cloud(
    #     '/home/ljl/ASC_Lidar_human/tmp/1/vertices.pcd', mocap_data.smpl_vertices(1))

    # coarse_align(mocap_data.smpl_vertices(1), lidar_points)
    # coarse_align(mocap_data.worldpos(1), mocap_data.pose(1), lidar_points)

    beta = refine_shape(mocap_data, 1, lidar_points)
    print(beta)
    pcd.save_point_cloud('/home/ljl/ASC_Lidar_human/tmp/refined.pcd',
                         transformation.affine(mocap_data.smpl_vertices(1, beta), transf))
