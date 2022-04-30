import numpy as np
import os
import shutil
import argparse

def filter_tracking_by_id(folder, ids):
    pcds = os.listdir(folder)
    os.makedirs(folder + '_slect', exist_ok=True)
    for pcd_path in pcds:
        if os.path.isdir(pcd_path):
            continue
        if pcd_path.endswith('.pcd') and int(pcd_path.split('_')[0]) in ids:
            shutil.copyfile(os.path.join(folder, pcd_path), os.path.join(folder + '_slect', pcd_path))
            print(f'{pcd_path} saved in {folder}_slect')

ids = [2,303,216,421,733,832,1037,1207,3437,3116,3218,4753,4922,5222,5725
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=str,
                        help='A directory', default="C:\\Users\\Yudi Dai\\segment_by_tracking")
    args, opts = parser.parse_known_args()
    filter_tracking_by_id(args.folder, ids)

