import open3d as o3d
import os
import numpy as np
import time
import cv2
import sys

view = {
    
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : 'false',
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 21.234885999999999, 8.7522749999999991, 4.0378610000000004 ],
			"boundingbox_min" : [ -16.570497, -43.161940999999999, -11.787841 ],
			"field_of_view" : 60.0,
			"front" : [ -0.21264672737025894, 0.8265871062578114, 0.52109032336731853 ],
			"lookat" : [ 3.5715111529651553, -13.836228448141792, -8.448559395394982 ],
			"up" : [ 0.059754191656269703, -0.52128847820400914, 0.85128594436373373 ],
			"zoom" : 0.21999999999999981
		}
	],
	"version_major" : 1,
	"version_minor" : 0
} # 近处的视角

lidar_cap_view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : 'false',
	"trajectory" : 
	[
		{
			"boundingbox_max": [89.155227661132813, 181.16168212890625, 12.757795333862305],
			"boundingbox_min": [-147.53880310058594, -169.27891540527344, -9.2261753082275391],
			"field_of_view": 60.0,
			"front": [-0.58479944824989349, -0.79081211293537867, 0.18062615359161649],
			"lookat": [16.50753367841779, 23.681048876301762, -4.7104662625614928],
			"up": [0.10862642610751434, 0.14432037550700175, 0.98355067422305531],
			"zoom": 0.099999999999999728
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

# lidar_cap_view = {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : 'false',
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 26.94597053527832, 58.301792144775391, 3.1996345520019531 ],
# 			"boundingbox_min" : [ -0.059999999999999998, -4.8996920585632324, -5.0722103118896484 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.095937206915887974, -0.9522310181928646, 0.28991747156830455 ],
# 			"lookat" : [ 16.50753367841779, 23.681048876301762, -4.7104662625614928 ],
# 			"up" : [ 0.01159309960734076, 0.29017227131180834, 0.95690420262596865 ],
# 			"zoom" : 0.27999999999999969
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# } # lidarcap 近处的视角

def visulize_point_clouds(file_path):
    files     = sorted(os.listdir(file_path))
    vis        = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Mesh vis', width=1280, height=720)
    pointcloud = o3d.geometry.PointCloud()
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    to_reset = True
    vis.add_geometry(pointcloud, reset_bounding_box=False)
    vis.add_geometry(axis_pcd, reset_bounding_box=False)
    ctr = vis.get_view_control()

    for i, file_name in enumerate(files):
        if i < 150:
            continue
        if file_name.endswith('.txt'):
            pts = np.loadtxt(os.path.join(file_path, file_name))
            pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3])  
        elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(os.path.join(file_path, file_name))
            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors
        else:
            continue
            

        vis.update_geometry(pointcloud)
        if to_reset:
            
            vis.reset_view_point(True)
            ctr.set_up(np.array(lidar_cap_view['trajectory'][0]['up']))
            ctr.set_lookat(np.array(lidar_cap_view['trajectory'][0]['lookat']))
            ctr.set_front(np.array(lidar_cap_view['trajectory'][0]['front']))
            ctr.set_zoom(lidar_cap_view['trajectory'][0]['zoom'])
            to_reset = False
        vis.poll_events()
        vis.update_renderer()
        cv2.waitKey(10)
        if i>= 150:
            out_dir = os.path.join(path, 'imgs')
            os.makedirs(out_dir, exist_ok=True)
            outname = os.path.join(out_dir, '{:04d}.jpg'.format(i-150))
            vis.capture_screen_image(outname)
    while True:
        vis.poll_events()
        vis.update_renderer()
        cv2.waitKey(10)

        # if DESTROY:
        #     vis.destroy_window()
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Wrong!')
        exit()
    # path = "G:\\Human_motion\\1101\\1101rockclimbing000_frames"
    # path = "F:\\segments"
    path = sys.argv[1]
    visulize_point_clouds(path)
