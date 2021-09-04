from collections import Counter
from sklearn.cluster import DBSCAN

import numpy as np
import os
import pcl

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def crop_points(points, crop_box):
    x_min, y_min = crop_box['min']
    x_max, y_max = crop_box['max']
    mask = np.logical_and(points[:, 0] > x_min, points[:, 1] > y_min)
    mask = np.logical_and(mask, points[:, 0] < x_max)
    mask = np.logical_and(mask, points[:, 1] < y_max)
    return points[mask].copy()


def pcap_to_pcds(pcap_path, pcds_dir):
    assert os.path.isabs(pcap_path) and os.path.isfile(pcap_path)
    assert os.path.isabs(pcds_dir) and os.path.isdir(pcds_dir)
    read_pcap_bin_path = os.path.join(ROOT_PATH, 'bin', 'read_pcap')
    os.system(
        '{} --in_file {} --out_dir {}'.format(read_pcap_bin_path, pcap_path, pcds_dir))


def dbscan_outlier_removal(points):
    labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(points)
    sort_labels = [x[0] for x in Counter(labels).most_common()]
    if -1 in sort_labels:
        sort_labels.remove(-1)
    if len(sort_labels) == 0:
        return points[0].reshape(1, -1)
    return points[labels == sort_labels[0]]


def erase_background(points, bg_kdtree):
    EPSILON = 0.12
    EPSILON2 = EPSILON ** 2
    squared_distance = bg_kdtree.nearest_k_search_for_cloud(
        pcl.PointCloud(points), 1)[1].flatten()
    erased_points = points[squared_distance > EPSILON2]
    if erased_points.shape[0] == 0:
        erased_points = points[0].reshape(1, -1)
    # return erased_points
    return dbscan_outlier_removal(erased_points)


def get_kdtree(points):
    return pcl.PointCloud(points.astype(np.float32)).make_kdtree_flann()


if __name__ == '__main__':
    kdtree = get_kdtree(np.random.random((1000, 3)))
    print(kdtree)
