import numpy as np
import pickle as pkl
import os

def read_pkl_file(file_name):
    """
    Reads a pickle file
    Args:
        file_name:
    Returns:
    """
    with open(file_name, 'rb') as f:
       data = pkl.load(f)
    return data

def save_trans(filepath):
    results = read_pkl_file(filepath)
    trans = results['transes']
    np.savetxt(filepath.split('.')[0]+'.txt', trans)
    print('save: ', filepath.split('.')[0]+'.txt')

hps = 'E:\\SCSC_DATA\\HumanMotion\\HPS\\result'
results_list = os.listdir(hps)
for r in results_list:
    rf = os.path.join(hps, r)
    if r.split('.')[-1] == 'pkl':
        save_trans(rf)    