from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import sys
import os


def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('save traj in: ', save_file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('请输入 [*lidar_traj.txt]')
        lidar_file = "/media/daiyudi/3T/SCSC_DATA/HumanMotion/0913/001/20210913daiyudi01_pcap_to_txt_2930_to_3000/traj_with_timestamp_new1.txt"
    else:
        lidar_file = sys.argv[1]
    # 1. 读取数据
    lidar = np.loadtxt(lidar_file, dtype=float)

    # 2. 拟合旋转量
    # Spherical Linear Interpolation of Rotations.
    ##
    frame_time = 0.05
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import RotationSpline
    times = lidar[:, -1].astype(np.float64)
    time_gap = times[1:] - times[:-1] 
    fit_list = np.arange(times.shape[0] - 1)[time_gap > frame_time * 1.5] # 找到间隔超过1.5倍的单帧时间

    fit_time = []
    for i, t in enumerate(times):
        fit_time.append(t)
        if i in fit_list:
            for j in range(int(time_gap[i]//frame_time)):
                if time_gap[i] - j * frame_time >= frame_time * 1.5:
                    fit_time.append(fit_time[-1] + frame_time)
    fit_time = np.asarray(fit_time)
    rotations = R.from_quat(lidar[:, 4: 8])
    spline = RotationSpline(times, rotations)
    quat_plot = spline(fit_time).as_quat()

    trajs = lidar[:, 1:4]  # 待拟合轨迹
    trajs_plot = []  # 拟合后轨迹
    segment = 10
    for i in range(0, lidar.shape[0], segment):
        s = i-1   # start index
        e = i+segment   # end index
        if lidar.shape[0] < e:
            s = lidar.shape[0] - segment
            e = lidar.shape[0]
        if s < 0:
            s = 0

        ps = s - segment//2 
        pe = e + segment//2
        if ps < 0:
            ps = 0
            pe += segment//2
        if pe > lidar.shape[0]:
            ps -= segment//2
            pe = lidar.shape[0]
        
        fp = np.polyfit(times[ps:pe],
                        trajs[ps:pe], 4)  # 分段拟合轨迹
        if s == 0:
            fs = np.where(fit_time == times[0])[0][0] # 拟合轨迹到起始坐标
        else:
            fs = np.where(fit_time == times[i - 1])[0][0] # 拟合轨迹到起始坐标

        fe = np.where(fit_time == times[e-1])[0][0] # 拟合轨迹到结束坐标

        if e == lidar.shape[0]:
            fe += 1
        for j in fit_time[fs: fe]:
            trajs_plot.append(np.polyval(fp, j))

    trajs_plot = np.asarray(trajs_plot)
    # for i in range(lidar.shape[0]):
    #     # 保留原来的轨迹
    #     trajs_plot[times[i] - times[0]] = trajs[i]
    frame_id = np.arange(trajs_plot.shape[0]).astype(np.int64) + int(lidar[0,0])
    lidar = np.concatenate(
        (frame_id.reshape(-1, 1), trajs_plot, quat_plot, fit_time.reshape(-1,1)), axis=1)

    # 4. 保存轨迹
    save_in_same_dir(lidar_file, lidar, '_afterFit')  # 保存有效轨迹
