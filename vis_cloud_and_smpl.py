from tkinter.messagebox import NO
import numpy as np
import pickle as pkl
import os
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# from sympy import re
from o3dvis import o3dvis
import matplotlib.pyplot as plt
from o3dvis import read_pcd_from_server, client_server, list_dir_remote

# 1. 载入pkl结果

# 2. 
def load_all_files_id(folder):
    results = os.listdir(folder)
    files_by_framid = {}
    files_by_humanid = {}
    for f in results:
        basename = f.split('.')[0]
        humanid = basename.split('_')[0]
        frameid = basename.split('_')[1]
        if frameid in files_by_framid:
            files_by_framid[frameid].append(humanid)
        else:
            files_by_framid[frameid] = [humanid]

        if humanid in files_by_humanid:
            files_by_framid[humanid].append(frameid)
        else:
            files_by_framid[humanid] = [frameid]
    return files_by_framid, files_by_humanid

