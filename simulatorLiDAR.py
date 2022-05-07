import os
import open3d as o3d
import numpy as np
from glob import glob
import sys

def hidden_point_removal(pcd, camera = [0, 0, 0]):
    diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print("Define parameters used for hidden_point_removal")
    # camera = [view_point[0], view_point[0], diameter]
    # camera = view_point
    radius = diameter * 100

    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    return pcd


def select_points_on_the_scan_line(points, view_point=None, scans=64, line_num=1024, fov_up=16.2, fov_down=-16.2, precision=1.1):
    
    fov_up = np.deg2rad(fov_up)
    fov_down = np.deg2rad(fov_down)
    fov = abs(fov_down) + abs(fov_up)

    ratio = fov/(scans - 1)   # 64bins 的竖直分辨率
    hoz_ratio = 2 * np.pi / (line_num - 1)    # 64bins 的水平分辨率
    # precision * np.random.randn() 

    print(points.shape[0])
    if view_point is not None:
        points -= view_point
    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    
    # pc_ds = []

    saved_box = { s:{} for s in np.arange(scans)}

    #### 筛选fov范围内的点
    for idx in range(0, points.shape[0]):
        rule1 =  pitch[idx] >= fov_down
        rule2 =  pitch[idx] <= fov_up
        rule3 = abs(pitch[idx] % ratio - 0.0) < 0.005
        rule4 = abs(yaw[idx] % hoz_ratio - 0.0) < 0.0005
        if rule1 and rule2 and rule3 and rule4:
            scanid = int((pitch[idx] + 1e-4) // ratio + scans // 2)
            pointid = int((yaw[idx] + 1e-4) // hoz_ratio)

            if pointid > 0 and scan_x[idx] < 0:
                pointid += 1024 // 2
            elif pointid < 0 and scan_y[idx] < 0:
                pointid += 1024 // 2
            
            if pointid in saved_box[scanid]:
                if depth[idx] < saved_box[scanid][pointid]['depth']:
                    saved_box[scanid][pointid].update({'points': points[idx], 'depth': depth[idx]})
            else:
                saved_box[scanid][pointid] = {'points': points[idx], 'depth': depth[idx]}

    save_points  =[]
    for key, value in saved_box.items():
        if len(value) > 0:
            for k, v in value.items():
                save_points.append(v['points']) 

    # pc_ds = np.array(pc_ds)
    save_points = np.array(save_points)


    #####
    print(save_points.shape)
    pc=o3d.open3d.geometry.PointCloud()
    pc.points= o3d.open3d.utility.Vector3dVector(save_points)
    pc.paint_uniform_color([0.5, 0.5, 0.5])
    pc.estimate_normals()

    return pc

def simulatorLiDAR(root, out_root):
    
    out_root = root
    origin_folder = os.path.join(out_root, 'origin')
    ds_folder = os.path.join(out_root, 'ds')
    if not os.path.exists(origin_folder):
            print(f"Creating output folder: {origin_folder}")
            os.makedirs(out_root+'/origin')
    if not os.path.exists(ds_folder):
            print(f"Creating output folder: {ds_folder}")
            os.makedirs(ds_folder)        
    filelist = sorted(glob(root+'\\*.ply'))
    

    for index in range(0, len(filelist)):
        print(f'Process {filelist[index]}')
        point_clouds = o3d.open3d.io.read_triangle_mesh(filelist[index])
        if len(point_clouds.triangles) > 0:
            point_clouds.compute_vertex_normals()
            pcd = point_clouds.sample_points_poisson_disk(100000)
        else:
            pcd = o3d.io.read_point_cloud(filelist[index])
        #     pcd = point_clouds

        # point_clouds
        view_point = point_clouds.get_center()
        view_point[0] += 0
        view_point[1] += -6.0
        view_point[2] += 0

        pc_ds = select_points_on_the_scan_line(np.asarray(pcd.points))
        # pc_ds = select_points_on_the_scan_line(np.asarray(pcd.points), view_point)
        
        filename, _ = os.path.splitext(os.path.basename(filelist[index]))
        
        # save_path = os.path.join(ds_folder, filename + '_origin.pcd')
        save_path = os.path.join(ds_folder, filename + '.pcd')

        # o3d.io.write_point_cloud(save_path, pcd)
        o3d.io.write_point_cloud(save_path, pc_ds)

if __name__ == '__main__':

    simulatorLiDAR('C:\\Users\\DAI\\Desktop\\temp\\segment_by_tracking_03_straight\\0013', '')
