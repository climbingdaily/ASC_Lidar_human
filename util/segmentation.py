import random
import open3d as o3d
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
import math
PAUSE = False

class Segmentation():
    def __init__(self, pcd):
        self.pcd = pcd
        self.segment_models = {}
        self.segments = {}
        self.segments_idx = {}

    def set_pcd(self, pcd):
        self.pcd = pcd
        self.segments.clear()
        self.segment_models.clear()
        self.segments_idx.clear()

    def seg_by_normals(self, eps = 0.15):
        """
        eps (float):  Density parameter that is used to find neighbouring points.
        """
        # normal_segments = {}
        normal_segments_idx = {}
        normal_pcd = o3d.geometry.PointCloud()
        normal_pcd.points = self.pcd.normals
        points_number = np.asarray(self.pcd.points).shape[0]
        lables = np.array(normal_pcd.cluster_dbscan(eps=0.15, min_points=200))
        if len(lables) > 100:
            for i in range(lables.max() + 1):
                normal_segments_idx[i] = list(np.where(lables == i)[0])
            rest_idx = list(np.where(lables == -1)[0])
            return normal_segments_idx, rest_idx
        else:
            return [], lables.shape[0]

    def loop_ransac(self, seg, count = 0, max_plane_idx=5, distance_threshold=0.02):
        rest = seg
        plane_count = 0
        for i in range(max_plane_idx):
            colors = plt.get_cmap("tab20")(i + count)
            points_number = np.asarray(rest.points).shape[0]
            if points_number < 50:
                break

            self.segment_models[i + count], inliers = rest.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=1200)

            if len(inliers) < 50:
                break

            self.segments[i + count] = rest.select_by_index(inliers)
            self.segments[i + count].paint_uniform_color(list(colors[:3]))
            plane_count += 1
            rest = rest.select_by_index(inliers, invert=True)
            # print("pass", i, "/", max_plane_idx, "done.")
        return rest, plane_count
        

    def dbscan_with_ransac(self, seg, count, max_plane_idx=5, distance_threshold=0.02):
        rest_idx = seg
        d_threshold = distance_threshold * 10
        plane_count = 0

        for i in range(max_plane_idx):
            if len(rest_idx) < 100:
                break
            rest = self.pcd.select_by_index(rest_idx)
            colors = plt.get_cmap("tab20")(count + i)
            
            points_number = np.asarray(rest.points).shape[0]
            if points_number < 20:
                break

            equation, inliers = rest.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)

            if len(inliers) < 20:
                break
            
            self.segment_models[i + count] = equation
            self.segments[i + count] = rest.select_by_index(inliers)

            labels = np.array(self.segments[i + count].cluster_dbscan(eps=d_threshold, min_points=15))
            candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
            if len(labels) <= 0:
                print('======================')
                print('======Catch u=========')
                print('======================')
            best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0][0]])
            # print("the best candidate is: ", best_candidate)
            inlier_valid_idx = np.where(labels == best_candidate)[0]
            self.segments_idx[i + count] = [rest_idx[yy] for yy in [inliers[xx] for xx in inlier_valid_idx]]
            self.segments[i + count] = self.pcd.select_by_index(self.segments_idx[i + count])
            self.segments[i + count].paint_uniform_color(list(colors[:3]))

            rest_idx = list(set(rest_idx) - set(self.segments_idx[i + count])) # 
            rest_idx.sort()   # F*******************K!!!!!!!!!!!!!!!!!!!!!!!!!!!
            plane_count += 1

            # print("pass", i+1, "/", max_plane_idx, "done.")
        return rest_idx, plane_count

    def filter_plane(self):
        equations = []
        planes = []
        planes_idx = []
        rest_idx = []
        for i in range(len(self.segments)):
            pts_normals = np.asarray(self.segments[i].normals) 
            points_number = pts_normals.shape[0]
            error_degrees = abs(np.dot(pts_normals, self.segment_models[i][:3]))
            # error_degrees < cos20??(0.93969)
            if np.sum(error_degrees < 0.9063) > points_number * 0.25 or points_number < 150:
                # print('Max error degrer???', math.acos(min(error_degrees)) * 180 / math.pi)
                # print(f'More than {(np.sum(error_degrees < 0.93969)/(points_number/100)):.2f}%, sum points {points_number}')
                rest_idx += self.segments_idx[i]
            else:
                equations.append(self.segment_models[i])
                planes.append(self.segments[i])
                planes_idx.append(self.segments_idx[i])
        
        skip_idx = []
        new_plane_idx = []
        
        for i in range(len(planes)):
            if i in set(skip_idx):
                continue
            new_plane_idx.append([i])
                
            for j in range(i+1, len(planes)):
                pi = np.asarray(planes[i].points).mean(axis=0)
                pj = np.asarray(planes[j].points).mean(axis=0)
                distance = np.linalg.norm(pi - pj)
                # mean_cos = (abs(np.dot(equations[i][:3], pi-pj)) + abs(np.dot(equations[j][:3], pi-pj)))/2

                # project pi to plane j
                p_to_plane_dist = np.linalg.norm(np.dot(equations[j], np.concatenate((pi, np.ones(1)))) * equations[j][:3])

                # If two planes' normals smaller than 5?? and in the same height
                # We think they are in the same plane
                if abs(np.dot(equations[i][:3], equations[j][:3])) > 0.996:
                    if distance < 0.05 or p_to_plane_dist < 0.03:
                        # We think they are the same plane
                        new_plane_idx[-1].append(j)
                        skip_idx.append(j)
        new_eqs = []
        new_planes = []
        new_planes_idx = []
        for i in range(len(new_plane_idx)):
            equation = np.asarray(equations)[new_plane_idx[i]]
            
            D = np.min(equation[:,3])
            equation = equation.mean(axis=0)
            equation[3] = D
            new_eqs.append(equation)

            if len(new_plane_idx[i]) > 1:
                for idx in new_plane_idx[i]:
                    planes[new_plane_idx[i][0]] += planes[idx]
                    planes_idx[new_plane_idx[i][0]] += planes_idx[idx]
                    planes[new_plane_idx[i][0]].paint_uniform_color(np.asarray(planes[new_plane_idx[i][0]].colors)[0])
            new_planes.append(planes[new_plane_idx[i][0]])
            new_planes_idx.append(planes_idx[new_plane_idx[i][0]])
        
        return new_eqs, new_planes, new_planes_idx, rest_idx,       

    def dbscan(self, rest):
        labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=10))
        max_label = labels.max()
        for i in range(max_label):
            list[np.where(labels == i)[0]]
        print(f"point cloud has {max_label + 1} clusters")

        colors = plt.get_cmap("tab10")(
            labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

    def run(self, max_plane = 5, distance_thresh = 0.02):
        normal_segments_idx, sum_rest_idx = self.seg_by_normals()
        count = 0
        if len(normal_segments_idx) > 0:
            for i in range(len(normal_segments_idx)):
                # z_value = np.asarray(self.pcd.select_by_index(normal_segments_idx[i]).normals).mean(axis=0)[2]
                # rest, plane_count = self.loop_ransac(normal_segments_idx[i], count, max_plane, distance_thresh)
                rest_idx, plane_count = self.dbscan_with_ransac(normal_segments_idx[i], count, max_plane, distance_thresh)
                count += plane_count
                sum_rest_idx += rest_idx

        sum_rest_idx, plane_count = self.dbscan_with_ransac(sum_rest_idx, count, max_plane, distance_thresh)
        count += plane_count

        equations, planes, planes_idx, rest_plane = self.filter_plane()
        sum_rest_idx += rest_plane

        print(f'Sum plane: {len(planes)}')

        if len(planes) > 0:
            z_order = np.argsort(-np.asarray(equations)[:,2])
            equations = [equations[oo] for oo in z_order]
            planes = [planes[oo] for oo in z_order]
            planes_idx = [planes_idx[oo] for oo in z_order]

        return equations, planes, planes_idx, sum_rest_idx


PAUSE = False
DESTROY = False
REMOVE = False
ADD = True
def pause_callback(vis):
    global PAUSE
    PAUSE = not PAUSE
    print('Pause', PAUSE)
    return False

def destroy_callback(vis):
    global DESTROY
    DESTROY = not DESTROY
    return False

def remove_scene_geometry(vis):
    global REMOVE
    REMOVE = not REMOVE
    return False

def add_scene_geometry(vis):
    global ADD
    ADD = not ADD
    return False

# key_to_callback = {}
# key_to_callback[ord("K")] = change_background_to_black
# key_to_callback[ord("R")] = load_render_option
# key_to_callback[ord(",")] = capture_depth
# key_to_callback[ord(".")] = capture_image
# key_to_callback[ord("T")] = pause_vis
# o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def creat_vis():
    # vis = o3d.visualization.Visualizer()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(' '), add_scene_geometry)
    vis.register_key_callback(ord("D"), destroy_callback)
    # vis.register_key_callback(ord("V"), remove_scene_geometry)
    # vis.register_key_callback(ord("A"), add_scene_geometry)
    # vis.register_key_callback(ord('A'), o3d_callback_rotate)
    # vis.create_window(window_name='RT', width=1920, height=1080)
    vis.create_window(window_name='RT', width=1280, height=720)
    return vis


if __name__ == "__main__":
    Vector3dVector = o3d.utility.Vector3dVector
    # import pyransac3d as pyrsc
    vis = creat_vis()
    idx = 760
    # pcd_file = f"E:\\SCSC_DATA\HumanMotion\\1023\\ouster_test.pcd"
    pcd_file = f"E:\\SCSC_DATA\HumanMotion\\1022\\SMPL\\rockclimbing_step_1_grid\\grid_{idx}.pcd"
    smpl_fle = f"E:\\SCSC_DATA\HumanMotion\\1022\\SMPL\\rockclimbing_step_1\\{idx}_smpl.ply"

    smpl = o3d.io.read_triangle_mesh(smpl_fle)
    smpl.compute_vertex_normals()
    smpl.paint_uniform_color([75/255, 145/255, 183/255])
    vis.add_geometry(smpl)

    pcd = o3d.io.read_point_cloud(pcd_file)
    scene_pcd = pcd.voxel_down_sample(voxel_size=0.04)
    scene_points = np.asarray(scene_pcd.points)
    scene_colors = np.asarray(scene_pcd.colors)
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=100))
    scene_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # ==============================================
    # Ransac ????????????
    # ==============================================
    seg_pcd = Segmentation(scene_pcd)
    plane_equations, segments, segments_idx, rest = seg_pcd.run(10, 0.01)
    # for i in range(len(segments)):
        # vis.add_geometry(segments[i])
    # vis.add_geometry(rest)
    p_count = len(segments)
    idx = 0
    reset_view = True
    while True:
            # o3dcallback()
        vis.poll_events()
        vis.update_renderer()
        cv2.waitKey(10)
        if DESTROY:
            vis.destroy_window()
        if ADD:
            if idx > 0:
                reset_view = False
            if idx < p_count:
                cc = scene_pcd.select_by_index(segments_idx[idx])
                vis.add_geometry(cc, reset_bounding_box=reset_view)
                # vis.add_geometry(segments[idx], reset_bounding_box=reset_view)
                print(f'Add num {idx + 1} plane.', plane_equations[idx])
                print(cc, 'center: ', np.asarray(cc.points).mean(axis=0))
                idx +=1
            # elif idx == p_count:
            #     vis.add_geometry(rest, reset_bounding_box=reset_view)
            #     print(f'Add rest points.')
            else:
                print('No more planes')
            ADD = False
