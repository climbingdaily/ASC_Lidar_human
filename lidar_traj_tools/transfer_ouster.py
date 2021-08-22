import pandas as pd  
import sys
import numpy as np
import os
from pathlib import Path

def save_in_same_dir(file_dir, file_path, data):
    file_name = Path(file_path).stem
    file_name = file_name.split('Frame ')[-1][:-1]
    save_file = os.path.join(file_dir, file_name + '.txt')
    np.savetxt(save_file, data, fmt='%.6f')
    print('Save frame in: ', save_file)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('请输入 [*.csv]')
        csvdir = "E:\\SCSC_DATA\HumanMotion\\0821\\20210821daoyudi2"
    else:
        csvdir = sys.argv[1]
    csvfiles = os.listdir(csvdir)
    dirname = csvdir + '_txt'
    os.makedirs(dirname, exist_ok=True)

    for csv in csvfiles:
        frame_data_csv=pd.read_csv(os.path.join(csvdir, csv), dtype=np.float32)
        frame_data = np.asarray(frame_data_csv)
        valid = []
        for i in range(frame_data.shape[0]):
            # range < 0.1m
            if frame_data[i, 7] < 100:
                continue
            valid.append(i)
        if len(valid) > 20000:
            save_frame = frame_data[valid][:, [0, 1, 2, 4, 8, 5]]   #Point:0 Point:1 Point:2 Reflectivity Raw Timestamp Channel
            save_frame[:, 4] = save_frame[:, 4] / 1e9       # nano second -> second
            order = np.argsort(save_frame[:, 4]).tolist()      # 以时间进行排序
            save_frame = save_frame[order] 
            save_in_same_dir(dirname, csv, save_frame)