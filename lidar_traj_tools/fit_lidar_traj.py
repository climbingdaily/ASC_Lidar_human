from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import sys
import os
import configargparse


def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f',
                  '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('save traj in: ', save_file)


def filterTraj(lidar_file, frame_time=0.05, segment=20 , dist_thresh=0.03, save_type='b'):
    # 1. 读取数据
    lidar = np.loadtxt(lidar_file, dtype=float)

    # 2. Spherical Linear Interpolation of Rotations.
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import RotationSpline
    times = lidar[:, -1].astype(np.float64)
    time_gap = times[1:] - times[:-1]
    fit_list = np.arange(
        times.shape[0] - 1)[time_gap > frame_time * 1.5]  # 找到间隔超过1.5倍的单帧时间

    fit_time = []
    for i, t in enumerate(times):
        fit_time.append(t)
        if i in fit_list:
            for j in range(int(time_gap[i]//frame_time)):
                if time_gap[i] - j * frame_time >= frame_time * 1.5:
                    fit_time.append(fit_time[-1] + frame_time)
    fit_time = np.asarray(fit_time)
    R_quats = R.from_quat(lidar[:, 4: 8])
    quats = R_quats.as_quat()
    spline = RotationSpline(times, R_quats)
    quats_plot = spline(fit_time).as_quat()

    trajs = lidar[:, 1:4]  # 待拟合轨迹
    trajs_plot = []  # 拟合后轨迹
    for i in range(0, lidar.shape[0], segment):
        s = i-1   # start index
        e = i+segment   # end index
        if lidar.shape[0] < e:
            s = lidar.shape[0] - segment
            e = lidar.shape[0]
        if s < 0:
            s = 0

        ps = s - segment//2  # filter start index
        pe = e + segment//2  # # filter end index
        if ps < 0:
            ps = 0
            pe += segment//2
        if pe > lidar.shape[0]:
            ps -= segment//2
            pe = lidar.shape[0]

        fp = np.polyfit(times[ps:pe],
                        trajs[ps:pe], 3)  # 分段拟合轨迹
        if s == 0:
            fs = np.where(fit_time == times[0])[0][0]  # 拟合轨迹到起始坐标
        else:
            fs = np.where(fit_time == times[i - 1])[0][0]  # 拟合轨迹到起始坐标

        fe = np.where(fit_time == times[e-1])[0][0]  # 拟合轨迹到结束坐标

        if e == lidar.shape[0]:
            fe += 1
        for j in fit_time[fs: fe]:
            trajs_plot.append(np.polyval(fp, j))

    trajs_plot = np.asarray(trajs_plot)
    frame_id = -1 * np.ones(trajs_plot.shape[0]).reshape(-1, 1)
    valid_idx = []

    for i, t in enumerate(times):
        old_id = np.where(fit_time == t)[0][0]
        frame_id[old_id] = lidar[i, 0]
        if np.linalg.norm(trajs_plot[old_id] - trajs[i]) < dist_thresh:
            # print(f'ID {i}, dist {np.linalg.norm(trajs_plot[old_id] - trajs[i]):.3f}')
            trajs_plot[old_id] = trajs[i]
            quats_plot[old_id] = quats[i]
            if save_type == 'a':
                valid_idx.append(old_id)

        if save_type == 'b':
            valid_idx.append(old_id)
    
    if save_type == 'c':
        valid_idx = np.arange(trajs_plot.shape[0]).astype(np.int64)

    fitLidar = np.concatenate(
        (frame_id[valid_idx], trajs_plot[valid_idx], quats_plot[valid_idx], fit_time[valid_idx].reshape(-1, 1)), axis=1)

    # 4. 保存轨迹
    save_in_same_dir(lidar_file, fitLidar, '_filt')  # 保存有效轨迹
    return fitLidar


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("-F", "--traj_file", type=str,
                        default="e:\\SCSC_DATA\HumanMotion\\1023\\shiyanlou002_lidar_trajectory.txt")
    parser.add_argument("-T", "--frame_time", type=float, default=0.05)
    parser.add_argument("-D", "--dist_thresh", type=float, default=0.03, help='轨迹与拟合值之间的距离阈值，>阈值则认为为离群点，保留拟合值')
    parser.add_argument("-S", "--save_type", type=str, default='b', help='a: 仅保留非离群点 | b: 都保留 | c: 额外保留时间插值的点')

    args = parser.parse_args()
    print('Filtering...')
    print(args)

    frame_time = args.frame_time
    # 用20个点拟合曲线，只优化中间10个点，每次滑动的窗口为10
    filter_window = int(0.5/frame_time)

    fitLidar = filterTraj(args.traj_file, frame_time=frame_time,
                          segment=filter_window, dist_thresh=args.dist_thresh, save_type = args.save_type)
