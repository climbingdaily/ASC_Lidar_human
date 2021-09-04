import cv2
import math
import os


def video_to_images(video_path, images_dir):
    assert os.path.isabs(video_path) and os.path.isfile(video_path)
    assert os.path.isabs(images_dir) and os.path.isdir(images_dir)
    os.system(
        'ffmpeg -i {} -r 30 -f image2 -v error -s 1920x1080 {}/%06d.jpg'.format(video_path, images_dir))


def project_points_on_image(points, img_filename, out_img_filename):
    img = cv2.imread(img_filename)
    for x, y in points:
        x = int(math.floor(x))
        y = int(math.floor(y))
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            continue
        cv2.circle(img, (x, y), 1, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(out_img_filename, img)
