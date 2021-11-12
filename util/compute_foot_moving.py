import numpy as np
import pandas as pd
import sys
import os
import configargparse
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def load_data(pospath):
    pos_data_csv = pd.read_csv(pospath, dtype=np.float32)

    pos_data = np.asarray(pos_data_csv) / 100  # cm -> m
    mocap_length = pos_data.shape[0]
    pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
    return pos_data


def detect_jump(left_foot, right_foot):
    lf_height = np.asarray(left_foot[:50]).mean()
    rf_height = np.asarray(right_foot[:50]).mean()
    left_foot = np.asarray(left_foot- lf_height)
    right_foot = np.asarray(right_foot- rf_height)
    l_peaks, lprop = find_peaks(left_foot, distance=80, height=0.05, prominence=0.05)
    r_peaks, rprop = find_peaks(right_foot, distance=80, height=0.05, prominence=0.05)
    jumps = []
    j = 0
    for i, lp in enumerate(l_peaks):
        if j >= len(r_peaks):
            break

        while r_peaks[j] < lp and abs(r_peaks[j] - lp) >= 5:
            if j+1 < len(r_peaks):
                j += 1
            else:
                break

        # two peaks at the same time(< 0.05s), distance < 0.05m, prominences > 0.2m
        if abs(r_peaks[j] - lp) < 5 :  
            peaks_dist = abs(lprop['peak_heights'][i] - rprop['peak_heights'][j])
            peak_prominences =  (lprop['prominences'][i] + rprop['prominences'][j])/2
            if peaks_dist < 0.05 and peak_prominences > 0.2:
                jumps.append(lp)
    return l_peaks, r_peaks, jumps

def detect_step_on_ground(left_foot, right_foot, distance=80, prominence=0.05):
    
    lf_height = np.asarray(left_foot[:50]).mean()
    rf_height = np.asarray(right_foot[:50]).mean()
    left_foot = np.asarray(left_foot- lf_height)
    right_foot = np.asarray(right_foot- rf_height)
    l_peaks, b = find_peaks(-left_foot, distance=distance, prominence=prominence)
    r_peaks, b = find_peaks(-right_foot, distance=distance, prominence=prominence)

    # plt.plot(right_foot, label='right foot')
    # plt.plot(left_foot, label='left_foot')
    # plt.scatter(l_peaks, left_foot[l_peaks], marker='o')
    # plt.scatter(r_peaks, right_foot[r_peaks], marker='x')
    # # plt.plot(left_foot[1:,1] - left_foot[:-1,1])
    # # plt.plot(right_foot[1:,1] - right_foot[:-1,1])
    # # plt.plot(pos_data[:,3,1] - pos_data[:,6,1])
    # plt.legend()
    # plt.show()
    return l_peaks, r_peaks

def plots(plt, right_foot, left_foot, label=''):
    l_peaks, r_peaks, jumps = detect_jump(left_foot, right_foot)
    lf_peaks, rf_peaks = detect_step_on_ground(left_foot, right_foot)
    
    lf_height = np.asarray(left_foot[:50]).mean()
    rf_height = np.asarray(right_foot[:50]).mean()
    # left_foot = np.asarray(left_foot- lf_height)
    # right_foot = np.asarray(right_foot- rf_height)

    plt.plot(right_foot, label='right'+label, linestyle = '--')
    plt.plot(left_foot, label='left'+label)
    plt.scatter(r_peaks, right_foot[r_peaks], marker='x')
    plt.scatter(l_peaks, left_foot[l_peaks], marker='o')
    plt.scatter(rf_peaks, right_foot[rf_peaks], marker='x')
    plt.scatter(lf_peaks, left_foot[lf_peaks], marker='o')
    print(jumps)

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("-P", "--pos_data", type=str,
                        default="e:\\SCSC_DATA\\HumanMotion\\1022\\rockclimbing_pos.csv")
    args = parser.parse_args()

    pos_data = load_data(args.pos_data)
    # left_foot = pos_data[:, 7, 1]
    # right_foot = pos_data[:, 3, 1]
    # plots(plt, right_foot, left_foot, label='foot')
    left_foot_end = pos_data[:, 8, 1]
    right_foot_end = pos_data[:, 4, 1]
    plots(plt, right_foot_end, left_foot_end, label='foot_end')

    right_hand = pos_data[:, 19, 1]
    left_hand = pos_data[:, 47, 1]
    plots(plt, right_hand, left_hand, label='hand')

    plt.legend()
    plt.show()

    print(pos_data.shape)
