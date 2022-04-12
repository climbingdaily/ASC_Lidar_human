import cv2
import sys
import os
from multiprocessing import Pool, Process
import functools
import numpy as np

def save_fig(frame_data):
    fig_save_path, frames = frame_data
    cv2.imwrite(fig_save_path, frames, [cv2.IMWRITE_JPEG_QUALITY, 1])
    print(f'\rSave img in {fig_save_path}', end="", flush=True)

def get_timestamp(vid_path, save_path = None):
    cameraCapture = cv2.VideoCapture(vid_path)

    success, frame = cameraCapture.read()
    if success:
        if save_path == None:
            save_path = os.path.join(os.path.dirname(
                vid_path), 'frames_' + os.path.basename(vid_path).split('.')[0])
        os.makedirs(save_path, exist_ok=True)
    save_frames = []
    time_stamps = []
    while success:
        if cv2.waitKey(1) == 27:
            break
        # cv2.imshow('Test camera', frame)
        success, frame = cameraCapture.read()
        milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
        time_stamps.append(milliseconds/1000)
        # save_frames.append(
        #     [os.path.join(save_path, f'{milliseconds/1000:3.3f}.jpg'), frame])
        cv2.imwrite(os.path.join(save_path, f'{milliseconds/1000:3.3f}.jpg'), frame)
        print(f'\r{milliseconds/1000:.3f}', end='', flush=True)
        # seconds = int(milliseconds//1000)
        # milliseconds = milliseconds%1000
        # minutes = 0
        # hours = 0
        # if seconds >= 60:
        #     minutes = seconds//60
        #     seconds = seconds % 60

        # if minutes >= 60:
        #     hours = minutes//60
        #     minutes = minutes % 60
        # print(f'{seconds}.{milliseconds:.0f}')
        # if len(save_frames) > 1000:
        #     with Pool(8) as p:
        #         p.map(functools.partial(save_fig), save_frames)
        #     print()
        #     save_frames.clear()

    cv2.destroyAllWindows()
    cameraCapture.release()
    if len(time_stamps) > 0:
        np.savetxt(os.path.join(save_path, 'time_stamp.txt'),
                   np.array(time_stamps), fmt='%.3f')
    # if success:
    #     with Pool(8) as p:
    #         p.map(functools.partial(save_fig), save_frames)
if __name__ == '__main__':
    vid_path = sys.argv[1]

    get_timestamp(vid_path)
