import os
import sys
from subprocess import run

# status = os.system(command)

def save_video(path):
    """输入文件夹图片所在文件夹, 利用ffmpeg生成视频

    Args:
        path ([str]): [图片所在文件夹]
    """            
    strs = path.split('\\')[-1]
    # strs = sys.argv[2]
    video_path = os.path.join(os.path.dirname(path), strs + '.avi')
    video_path2 = os.path.join(os.path.dirname(path), strs + '.mp4')

    command = f"ffmpeg -f image2 -i {path}\\{strs}_%4d.jpg -b:v 10M -c:v h264 -r 20  {video_path}"
    if not os.path.exists(video_path) and not os.path.exists(video_path2):
        run(command, shell=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Wrong!')
        exit()
    path = sys.argv[1]
    path_list = os.listdir(path)
    for p in path_list:
        if os.path.isdir(os.path.join(path, p)):
            save_video(os.path.join(path, p))