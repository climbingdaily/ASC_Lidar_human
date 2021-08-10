import cv2
from matplotlib.pyplot import axis
import numpy as np
import os
import math
from util import projection
from io3d import obj, pcd


def dai():
    # Dai
    intrinsic_matrix = np.array(
        [1515.72, 0, 1103.06,
         0, 1513.97, 632.042,
         0, 0, 1]
    ).reshape(3, 3)

    extrinsic_matrix = np.array([
        -0.927565, 0.36788, 0.065483, -1.18345,
        0.0171979, 0.217091, -0.976, -0.0448631,
        -0.373267, -0.904177, -0.207693, 8.36933,
        0, 0, 0, 1
    ]).reshape(4, 4)

    distortion_coefficients = np.array(
        [0.000953935, -0.0118572, 0.000438133, -0.000892954, 0.0208176])
    # Dai
    image_folder = 'data/02_image_after'
    img_filenames = sorted(os.listdir(image_folder),
                           key=lambda x: int(x[:-4]))[23:]
    os.makedirs('data/dai_out', exist_ok=True)
    point_clouds = np.loadtxt(
        'data/02_with_lidar_pos_cloud.txt')[:, :3].reshape(-1, 59, 3)[4:]
    print(point_clouds.shape, len(img_filenames))
    for i, img_filename in enumerate(img_filenames):
        img_filename = os.path.join(image_folder, img_filename)
        img = cv2.imread(img_filename)
        world_points = point_clouds[i]
        camera_points = projection.world_to_camera(
            world_points, extrinsic_matrix)
        pixel_points = projection.camera_to_pixel(
            camera_points, intrinsic_matrix, distortion_coefficients)
        for x, y in pixel_points:
            x = int(math.floor(x))
            y = int(math.floor(y))
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('data/dai_out/{}.jpg'.format(i + 1), img)
    exit()


def li():
    intrinsic_matrix = np.array([
        [1001.5891335, 0, 953.6128327],
        [0, 1000.9244526, 582.04816056],
        [0, 0, 1]])
    extrinsic_matrix = np.array([
        [-0.022845605917, -0.99949339352, 0.022159300388, -0.0026512677119],
        [-0.048575862249, -0.021029143708, -0.9985980977, 0.24298071602],
        [0.99855819254, -0.023889985732, -0.048070829873, 0.16285148859],
        [0, 0, 0, 1]])
    extrinsic_matrix = np.array([-0.029557344798, -0.99616517002, 0.082348754784, -0.017884169028, -0.019153438873, -
                                0.08180517669, -0.99646427876, 0.10070376894, 0.9993795621, -0.031030100107, -0.016662044878, 0.22480853278, 0, 0, 0, 1])
    extrinsic_matrix = extrinsic_matrix.reshape((4, 4))

    # extrinsic_matrix = np.array([
    #     [-9.9120421801525116e-01, -1.3089677551201390e-01,
    #         1.9499547413517729e-02, 1.5806716007560382e+02],
    #     [-1.3215123094734271e-01, 9.7108594514058533e-01, -
    #         1.9881684865603558e-01, -6.8613802009855448e+01],
    #     [7.0887479766656089e-03, -1.9964498819397472e-01, -
    #         9.7984265488962619e-01, 3.4967980284631008e+03],
    #     [0, 0, 0, 1]])

    distortion_coefficients = np.array([3.2083739041580001e-01,
                                        2.2269550643173597e-01,
                                        8.8895447057740762e-01,
                                        -2.8404775013002994e+00,
                                        4.867095044851689
                                        ])

    distortion_coefficients = np.zeros(5)

    # pc_prefix = 'data/2021-07-24-16-02-45-RS-0-Data-10000-12000'
    # pc_filenames = sorted(os.listdir(pc_prefix),
    #                       key=lambda x: int(x[:-4]))[535:]

    pc_prefix = 'data/point_cloud'
    pc_filenames = sorted(os.listdir(pc_prefix),
                          key=lambda x: int(x[:-4]))
    os.makedirs('data/0805', exist_ok=True)

    data = []

    for i, pc_filename in enumerate(pc_filenames):
        pc_filename = os.path.join(pc_prefix, pc_filename)
        # world_points = obj.read_point_cloud(pc_filename)
        world_points = pcd.read_point_cloud(pc_filename)[:, :3]
        camera_points = projection.world_to_camera(
            world_points, extrinsic_matrix)
        depth = camera_points[:, 2][:, None]

        pixel_points = projection.camera_to_pixel(
            camera_points, intrinsic_matrix, distortion_coefficients)
        data.append(np.concatenate((pixel_points, depth), axis=1))
        img_filename = 'data/image/{:05d}.jpg'.format(859 + i)
        img = cv2.imread(img_filename)
        for x, y in pixel_points:
            x = int(math.floor(x))
            y = int(math.floor(y))
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('data/0805/{}.jpg'.format(i + 1), img)
    data = np.array(data)
    np.save('data/depth.npy', data)


def zhang():

    from Robosense_object import Calibration

    calib = Calibration('1')
    pc_prefix = 'data/point_cloud'
    pc_filenames = sorted(os.listdir(pc_prefix),
                          key=lambda x: int(x[:-4]))

    # pc_prefix = 'data/2021-07-24-16-02-45-RS-0-Data-10000-12000'
    # pc_filenames = sorted(os.listdir(pc_prefix),
    #                       key=lambda x: int(x[:-4]))[535:]

    os.makedirs('data/0805', exist_ok=True)
    for i, pc_filename in enumerate(pc_filenames):
        pc_filename = os.path.join(pc_prefix, pc_filename)
        # world_points = obj.read_point_cloud(pc_filename)
        world_points = pcd.read_point_cloud(pc_filename)[:, :3]
        pixel_points = calib.project_robo_to_image(world_points)
        pixel_points = pixel_points[:, :2]
        img_filename = 'data/image/{:05d}.jpg'.format(859 + i)
        img = cv2.imread(img_filename)
        for x, y in pixel_points:
            x = int(math.floor(x))
            y = int(math.floor(y))
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('data/0805/{}.jpg'.format(i + 1), img)
        exit()


if __name__ == '__main__':
    dai()
