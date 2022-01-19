import open3d as o3d
import os
import numpy as np
import time
import cv2
import sys
from o3dvis import o3dvis

lidar_cap_view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : 'false',
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 14.945155143737793, 73.207733154296875, 3.0000000000000004 ],
			"boundingbox_min" : [ -17.210771560668945, -0.17999999999999999, -9.3062448501586914 ],
			"field_of_view" : 70.0,
			"front" : [ -0.1791689038167478, -0.94567240297038702, 0.27129727268315645 ],
			"lookat" : [ -1.8274709948062637, 32.785384934359762, -7.6441299152652933 ],
			"up" : [ 0.057188868300086702, 0.26528205271659239, 0.96247330656440888 ],
			"zoom" : 0.25999999999999984
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Wrong!')
        exit()
    path = sys.argv[1]
    vis = o3dvis()
    vis.visulize_point_clouds(path, skip=100, view = lidar_cap_view)
