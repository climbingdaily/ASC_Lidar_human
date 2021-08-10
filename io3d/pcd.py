import open3d as o3d
import numpy as np
import pcl


def read_point_cloud(filename):
    return np.asarray(pcl.load(filename))
