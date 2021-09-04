from collections import namedtuple
from concurrent.futures import as_completed, ThreadPoolExecutor
from io3d import mocap, pcd
from tqdm import tqdm
from typing import Dict, List
from util import mocap_util, path_util, pc_util, img_util, transformation

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import smpl.generate_ply

IMAGE_FRAME_RATE = 30
POINTCLOUD_FRAME_RATE = 10
MOCAP_FRAME_RATE = 90
MAX_THREAD_COUNT = 16

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s', datefmt='%b %d %H:%M:%S')
logger = logging.getLogger()


def dict_to_struct(d):
    return namedtuple('Struct', d.keys())(*d.values())


def prepare_dataset_dirs(dataset_prefix):
    subdir = {
        'root': ['calib', 'images', 'labels', 'pointclouds', 'mocaps'],
        'labels': ['2d', '3d'],
        '2d': ['mask', 'bbox', 'keypoints'],
        '3d': ['smpl', 'segmentation'],
        'smpl': ['param', 'mesh'],
    }

    dataset_dirs = {}
    import queue
    q = queue.Queue()
    q.put(('root', ''))
    while not q.empty():
        u, path = q.get()
        if u not in subdir:
            cur_path = os.path.join(dataset_prefix, path)
            dataset_dirs[u + '_dir'] = cur_path
            os.makedirs(cur_path, exist_ok=True)
        else:
            for v in subdir[u]:
                q.put((v, os.path.join(path, v)))
    return dataset_dirs


def prepare_current_dirs(raw_dir, dataset_dirs, index):
    cur_dirs = {'raw_dir': os.path.join(raw_dir, str(index))}
    for key, value in dataset_dirs.items():
        if key == 'calib_dir':
            cur_dirs[key] = value
        else:
            cur_dirs[key] = os.path.join(value, str(index))
        os.makedirs(cur_dirs[key], exist_ok=True)
    return dict_to_struct(cur_dirs)


def project(cur_dirs, mocap_data: mocap.MoCapData, bg_points, crop_box, mocap_indexes):
    img_dir = cur_dirs.images_dir
    pc_dir = cur_dirs.pointclouds_dir
    pc_projection_dir = cur_dirs.pc_projection_dir
    mocap_projection_dir = cur_dirs.mocap_projection_dir
    mocap_vertices_dir = cur_dirs.mocap_vertices_dir

    intrinsic_matrix = np.array([9.8430939171489899e+02, 0., 9.5851460160821068e+02, 0.,
                                 9.8519855566009164e+02, 5.8554990545554267e+02, 0., 0., 1.]).reshape(3, 3)
    extrinsic_matrix = np.array([0.0077827356937, -0.99995676791, 0.0050883521036, -0.0032019276082, 0.0016320898332, -0.0050757970921, -
                                0.99998578618, 0.049557315144, 0.99996838215, 0.007790929719, 0.0015925156874, 0.12791621362, 0, 0, 0, 1]).reshape((4, 4))

    distortion_coefficients = np.array([3.2083739041580001e-01,
                                        2.2269550643173597e-01,
                                        8.8895447057740762e-01,
                                        -2.8404775013002994e+00,
                                        4.867095044851689
                                        ])
    distortion_coefficients = np.zeros(5)

    C = np.array([[1.62427591e-02, -9.99862523e-01, 3.34354090e-03,
                   3.83774964e+01],
                  [9.65757161e-01, 1.65545404e-02, 2.58919002e-01,
                   -1.76110792e+01],
                  [-2.58938730e-01, -9.76512034e-04, 9.65893269e-01,
                   7.61942050e+00],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   1.00000000e+00]])

    pc_filenames = path_util.get_sorted_filenames_by_index(pc_dir)
    img_filenames = path_util.get_sorted_filenames_by_index(img_dir)
    bg_kdtree = pc_util.get_kdtree(pc_util.crop_points(bg_points, crop_box))

    lidar_point_clouds = []
    mocap_point_clouds = []
    mocap_to_lidar_translations = []

    first = True
    beta = np.array([-0.36286515, 2.5138147, 0.93359864, -0.06489615,
                     2.6256192, -1.2169701, 2.7481666, -1.7482048, -2.6938446, 2.7220988])
    for pc_filename, mocap_index in zip(pc_filenames, mocap_indexes):
        lidar_points = pcd.read_point_cloud(pc_filename)[:, :3]
        lidar_points = pc_util.crop_points(lidar_points, crop_box)
        lidar_points = pc_util.erase_background(lidar_points, bg_kdtree)
        mocap_points = mocap_data.smpl_vertices(mocap_index, beta=beta)
        # mocap_points = mocap_data.worldpos(mocap_index)
        # if first:
        #     first = False
        #     mocap_to_lidar_rotation = transformation.get_mocap_to_lidar_rotation(
        #         mocap_points, lidar_points, C)
        #     C = np.dot(np.linalg.inv(mocap_to_lidar_rotation), C)
        mocap_to_lidar_translation = transformation.get_mocap_to_lidar_translation(
            mocap_points, lidar_points, C) if lidar_points.shape[0] > 1 else None

        lidar_point_clouds.append(lidar_points)
        mocap_point_clouds.append(mocap_points)
        mocap_to_lidar_translations.append(mocap_to_lidar_translation)

    # smooth
    half_width = 10
    translation_sum = np.zeros((3, ))
    n = len(mocap_to_lidar_translations)
    l = 0
    r = 0
    cnt = 0
    aux = []
    for i in range(n):
        rb = min(n - 1, i + half_width)
        lb = max(0, i - half_width)
        while r <= rb:
            if mocap_to_lidar_translations[r] is not None:
                translation_sum += mocap_to_lidar_translations[r]
                cnt += 1
            r += 1
        while l < lb:
            if mocap_to_lidar_translations[l] is not None:
                translation_sum -= mocap_to_lidar_translations[l]
                cnt -= 1
            l += 1
        if cnt > 0:
            aux.append(translation_sum / cnt)
        else:
            aux.append(None)
    mocap_to_lidar_translations = aux

    index = 1
    for lidar_points, mocap_index, mocap_points, mocap_to_lidar_translation, img_filename in zip(lidar_point_clouds, mocap_indexes, mocap_point_clouds, mocap_to_lidar_translations, img_filenames):
        # lidar projection
        # pixel_points = transformation.camera_to_pixel(transformation.lidar_to_camera(
        #     lidar_points, extrinsic_matrix), intrinsic_matrix, distortion_coefficients)
        # pc_projection_filename = os.path.join(
        #     pc_projection_dir, os.path.basename(img_filename))
        # img_util.project_points_on_image(
        #     pixel_points, img_filename, pc_projection_filename)

        # mocap projection
        if mocap_to_lidar_translation is None:
            mocap_points = mocap_points[0].reshape(-1, 3)
        else:
            mocap_points = transformation.mocap_to_lidar(
                mocap_points, lidar_points, C, mocap_to_lidar_translation)

        smpl.generate_ply.save_ply(mocap_points, os.path.join(
            mocap_vertices_dir, '{:06d}.ply'.format(index)))

        index += 1
        # pixel_points = transformation.camera_to_pixel(transformation.lidar_to_camera(
        #     mocap_points, extrinsic_matrix), intrinsic_matrix, distortion_coefficients)
        # mocap_projection_filename = os.path.join(
        #     mocap_projection_dir, os.path.basename(img_filename))
        # img_util.project_points_on_image(
        #     pixel_points, img_filename, mocap_projection_filename)


def generate_keypoints(images_dir, keypoints_dir, openpose_path):
    # openpose
    origin_cwd = os.getcwd()
    openpose_bin_path = os.path.join(
        openpose_path, 'build/examples/openpose/openpose.bin')
    cmd = '{} --image_dir {} --write_json {} --display 0 --render_pose 0'.format(
        openpose_bin_path, images_dir, keypoints_dir)
    os.chdir(openpose_path)
    os.system(cmd)
    os.chdir(origin_cwd)


def generate_segmented_point_clouds(pc_dir: str,
                                    seg_dir: str,
                                    bg_points: np.ndarray,
                                    crop_box: np.ndarray):
    bg_kdtree = pc_util.get_kdtree(pc_util.crop_points(bg_points, crop_box))
    pc_filenames = path_util.get_sorted_filenames_by_index(pc_dir)

    def generate_segmented_point_cloud(pc_filename):
        lidar_points = pcd.read_point_cloud(pc_filename)[:, :3]
        lidar_points = pc_util.crop_points(lidar_points, crop_box)
        lidar_points = pc_util.erase_background(lidar_points, bg_kdtree)
        pcd.save_point_cloud(os.path.join(
            seg_dir, os.path.basename(pc_filename)), lidar_points)

    with ThreadPoolExecutor(MAX_THREAD_COUNT) as executor:
        tasks = [executor.submit(generate_segmented_point_cloud, pc_filename)
                 for pc_filename in pc_filenames]
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            pass
    print('Generate segmented point clouds finished')


def generate_pose(cur_dirs: Dict[str, str],
                  mocap_data: mocap.MoCapData,
                  mocap_indexes: List[int],
                  cur_process_info: Dict,
                  bg_points: np.ndarray):
    # TODO: C should be read for each instance
    C = np.array([[1.62427591e-02, -9.99862523e-01, 3.34354090e-03,
                 3.83774964e+01],
                  [9.65757161e-01, 1.65545404e-02, 2.58919002e-01,
                 -1.76110792e+01],
                  [-2.58938730e-01, -9.76512034e-04, 9.65893269e-01,
                 7.61942050e+00],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   1.00000000e+00]])

    pc_filenames = path_util.get_sorted_filenames_by_index(
        cur_dirs.pointclouds_dir)
    crop_box = cur_process_info['box']
    bg_kdtree = pc_util.get_kdtree(pc_util.crop_points(bg_points, crop_box))

    lidar_point_clouds = []
    mocap_point_clouds = []
    mocap_to_lidar_translations = []

    beta = np.array(cur_process_info['beta'])
    pose_and_betas = []
    print('Calculate MoCap to LiDAR translations')
    for pc_filename, mocap_index in tqdm(zip(pc_filenames, mocap_indexes)):
        lidar_points = pcd.read_point_cloud(pc_filename)[:, : 3]
        lidar_points = pc_util.crop_points(lidar_points, crop_box)
        lidar_points = pc_util.erase_background(lidar_points, bg_kdtree)
        # 72 + 10 (pose + beta)
        pose_and_betas.append(np.concatenate(
            (mocap_data.pose(mocap_index), beta)))
        mocap_points = mocap_data.smpl_vertices(mocap_index, beta=beta)
        mocap_to_lidar_translation = transformation.get_mocap_to_lidar_translation(
            mocap_points, lidar_points, C) if lidar_points.shape[0] > 1 else None

        lidar_point_clouds.append(lidar_points)
        mocap_point_clouds.append(mocap_points)
        mocap_to_lidar_translations.append(mocap_to_lidar_translation)
    np.save(os.path.join(cur_dirs.param_dir, 'param.npy'), pose_and_betas)

    # smooth
    half_width = 10
    translation_sum = np.zeros((3, ))
    n = len(mocap_to_lidar_translations)
    l = 0
    r = 0
    cnt = 0
    aux = []
    for i in range(n):
        rb = min(n - 1, i + half_width)
        lb = max(0, i - half_width)
        while r <= rb:
            if mocap_to_lidar_translations[r] is not None:
                translation_sum += mocap_to_lidar_translations[r]
                cnt += 1
            r += 1
        while l < lb:
            if mocap_to_lidar_translations[l] is not None:
                translation_sum -= mocap_to_lidar_translations[l]
                cnt -= 1
            l += 1
        if (mocap_to_lidar_translations[i] is not None) and (cnt > 0):
            aux.append(translation_sum / cnt)
        else:
            aux.append(None)
    mocap_to_lidar_translations = aux

    index = 1
    print('Saving Plys')
    for mocap_points, mocap_to_lidar_translation in tqdm(zip(mocap_point_clouds, mocap_to_lidar_translations)):
        if mocap_to_lidar_translation is None:
            mocap_points = mocap_points[0].reshape(-1, 3)
        else:
            mocap_points = transformation.mocap_to_lidar(
                mocap_points, C, translation=mocap_to_lidar_translation)
        smpl.generate_ply.save_ply(mocap_points, os.path.join(
            cur_dirs.mesh_dir, '{:06d}.ply'.format(index)))
        index += 1


def main(args):
    # parse args
    raw_dir = args.raw_dir
    dataset_dir = args.dataset_dir
    start_index = args.start_index
    end_index = args.end_index + 1  # the end_index input by user is closed endpoint
    openpose_path = args.openpose_path

    # parse generate params
    gen = {}
    for k, v in vars(args).items():
        if k.startswith('gen') and k != 'gen_all':
            gen[k[4:]] = v or args.gen_all
    gen = dict_to_struct(gen)
    if gen.mask:
        import mask_rcnn.inference

    dataset_dirs = prepare_dataset_dirs(dataset_dir)

    # read json
    with open(os.path.join(raw_dir, 'process_info.json')) as f:
        process_info = json.load(f)

    # read background points used for background subtraction
    bg_points = pcd.read_point_cloud(os.path.join(
        dataset_dirs['pointclouds_dir'], 'bg.pcd'))[:, :3]

    for index in range(start_index, end_index):
        print('index: ' + str(index))
        cur_dirs = prepare_current_dirs(raw_dir, dataset_dirs, index)
        video_path = path_util.get_one_path_by_suffix(cur_dirs.raw_dir, '.mp4')
        pcap_path = path_util.get_one_path_by_suffix(cur_dirs.raw_dir, '.pcap')
        bvh_path = path_util.get_one_path_by_suffix(cur_dirs.raw_dir, '.bvh')
        img_dir = cur_dirs.images_dir
        pc_dir = cur_dirs.pointclouds_dir
        mocap_dir = cur_dirs.mocaps_dir
        cur_process_info = process_info[str(index)]

        if args.gen_basic:
            img_util.video_to_images(video_path, img_dir)
            pc_util.pcap_to_pcds(pcap_path, pc_dir)
            mocap_util.get_csvs_from_bvh(bvh_path, mocap_dir)

            # 抽帧对齐
            img_frame_nums = len(os.listdir(img_dir))
            pc_frame_nums = len(os.listdir(pc_dir))
            mocap_frame_nums = pd.read_csv(
                path_util.get_one_path_by_suffix(mocap_dir, '_worldpos.csv')).shape[0]

            # 得到最多重叠的帧数
            pc_start_index = cur_process_info['start_index']['pointcloud']
            img_start_index = cur_process_info['start_index']['image']
            mocap_start_index = cur_process_info['start_index']['mocap']

            pc_indexes = np.arange(pc_start_index, pc_frame_nums + 1,
                                   POINTCLOUD_FRAME_RATE // POINTCLOUD_FRAME_RATE)
            img_indexes = np.arange(
                img_start_index, img_frame_nums + 1, IMAGE_FRAME_RATE // POINTCLOUD_FRAME_RATE)
            mocap_indexes = np.arange(
                mocap_start_index - 1, mocap_frame_nums, MOCAP_FRAME_RATE // POINTCLOUD_FRAME_RATE)
            n_frames = min(pc_indexes.shape[0], min(
                img_indexes.shape[0], mocap_indexes.shape[0]))

            pc_indexes = pc_indexes[:n_frames]
            img_indexes = img_indexes[:n_frames]
            mocap_indexes = mocap_indexes[:n_frames]

            path_util.rearrange_frames(img_dir, img_indexes)
            path_util.rearrange_frames(pc_dir, pc_indexes)
        else:
            mocap_frame_nums = pd.read_csv(
                path_util.get_one_path_by_suffix(mocap_dir, '_worldpos.csv')).shape[0]
            mocap_start_index = cur_process_info['start_index']['mocap']
            mocap_indexes = np.arange(
                mocap_start_index - 1, mocap_frame_nums, MOCAP_FRAME_RATE // POINTCLOUD_FRAME_RATE)
            n_frames = len(os.listdir(img_dir))
            mocap_indexes = mocap_indexes[:n_frames]

        if gen.keypoints:
            generate_keypoints(img_dir, cur_dirs.keypoints_dir, openpose_path)

        if gen.mask:
            mask_rcnn.inference.inference(
                img_dir, cur_dirs.bbox_dir, cur_dirs.mask_dir)

        if gen.seg_pc:
            generate_segmented_point_clouds(
                pc_dir, cur_dirs.segmentation_dir, bg_points, cur_process_info['box'])

        if gen.pose:
            worldpos_csv = path_util.get_one_path_by_suffix(
                mocap_dir, '_worldpos.csv')
            rotation_csv = path_util.get_one_path_by_suffix(
                mocap_dir, '_rotations.csv')
            mocap_data = mocap.MoCapData(worldpos_csv, rotation_csv)
            generate_pose(cur_dirs, mocap_data, mocap_indexes,
                          cur_process_info, bg_points)
        # project(cur_dirs, mocap.MoCapData(worldpos_csv, rotation_csv),
        #         bg_points, cur_process_info['box'], mocap_indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int, required=True)
    parser.add_argument('--end_index', type=int, required=True)

    parser.add_argument('--raw_dir', type=str, default='/xmu_gait/raw')
    parser.add_argument('--dataset_dir', type=str, default='/xmu_gait/dataset')
    parser.add_argument('--openpose_path', type=str,
                        default='/home/ljl/Tools/openpose')

    parser.add_argument('--log', action='store_true')

    parser.add_argument('--gen_all', action='store_true',
                        help='generate the images, point cloud files and csv files')
    parser.add_argument('--gen_basic', action='store_true')
    parser.add_argument('--gen_keypoints', action='store_true')
    parser.add_argument('--gen_mask', action='store_true')
    parser.add_argument('--gen_seg_pc', action='store_true')
    parser.add_argument('--gen_pose', action='store_true')
    # parser.add_argument('--write_smpl_vertices', action='store_true')

    logger.info('sdfsddfsdfsdfsfsdf')

    args = parser.parse_args()
    if not args.log:
        logger.disabled = True
    main(args)
