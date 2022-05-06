from locale import currency
import numpy as np
import sys 
import os
import shutil
import argparse
import open3d as o3d
from sqlalchemy import true
from sympy import O
from o3dvis import o3dvis
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vis_3d_box import load_data_remote
from o3dvis import list_dir_remote

def select_pcds_by_id(folder, ids):
    pcds = os.listdir(folder)
    os.makedirs(folder + '_slect', exist_ok=True)
    for pcd_path in pcds:
        if os.path.isdir(os.path.join(folder, pcd_path)):
            continue
        if pcd_path.endswith('.pcd') and int(pcd_path.split('_')[0]) in ids:
            shutil.copyfile(os.path.join(folder, pcd_path), os.path.join(folder + '_slect', pcd_path))
            print(f'{pcd_path} saved in {folder}_slect')

ids = [2,303,216,421,733,832,1037,1207,3437,3116,3218,4753,4922,5222,5725] # 0417-01
ids = [9, 152, 181, 193, 293, 379, 451, 674, 962, 1601, 1709, 2319, 2777, 1395, 2839, 92, 187, 275, 310, 1905, 1960] # 0417-03


class filter_tracking_by_interactive():
    def __init__(self, tracking_folder, remote=False):
        self.view_initialized = False
        self.vis = o3dvis(window_name='filter_tracking_by_interactive')
        self.checked_ids = {}
        self.real_person_id = []
        self.save_list = []
        self.pre_geometries = []
        self.pre_human_boxes = {}
        self.reID = {}
        self.tracking_folder = tracking_folder
        self.remote = remote
        self.load_data = load_data_remote(remote)
        self.join = '/' if remote else '\\'

    def copy_save_files(self):
        os.makedirs(self.tracking_folder + '_slect', exist_ok=True)
        for pcd_path in self.save_list:
            source = self.join.join(self.tracking_folder, pcd_path)
            if os.path.isdir(source):
                continue
            if pcd_path.endswith('.pcd'):
                humanid = pcd_path.split('_')[0]
                appendix = pcd_path.split('_')[1]
                target = self.join.join(self.tracking_folder + '_select', pcd_path)

                if humanid in self.reID:
                    humanid = self.reID[humanid]
                    target = self.join.join(os.path.dirname(
                        target), f'{humanid}_{appendix}')
                    # pts = self.load_data.load_point_cloud(source)
                    # pts = o3d.io.read_point_cloud(source)
                    # pts.paint_uniform_color(plt.get_cmap("tab20")(int(humanid) % 20)[:3])
                    # o3d.io.write_point_cloud(target, pts)
                # else:
                    # shutil.copyfile(source, target)

                _, stdout, _ = self.load_data.client.exec_command(f'cp {source} {target}')
                print(f'{pcd_path} saved in {self.tracking_folder}_slect')

    def add_box(self, box, color):
        transform = R.from_rotvec(
            box[6] * np.array([0, 0, 1])).as_matrix()
        center = box[:3]
        extend = box[3:6]
        bbox = o3d.geometry.OrientedBoundingBox(center, transform, extend)
        bbox.color = color
        self.vis.add_geometry(bbox, reset_bounding_box = False, waitKey=0)
        return bbox

    def interactive_choose(self, file_path = None, scene_path=None, pre_box=None, cur_box=None, strs='a real human'):
        if scene_path is not None:
            # pts = o3d.io.read_point_cloud(scene_path)
            pts = self.load_data.load_point_cloud(scene_path)
            pts.paint_uniform_color([0.5,0.5,0.5])
            if self.view_initialized:
                self.vis.add_geometry(pts, reset_bounding_box=False)
            else:
                self.view_initialized = True
                self.vis.add_geometry(pts, reset_bounding_box=True)
                self.vis.set_view()

            self.pre_geometries.append(pts)

        if file_path is not None:
            # pts = o3d.io.read_point_cloud(file_path)
            pts = self.load_data.load_point_cloud(file_path)
            
            self.vis.add_geometry(pts, reset_bounding_box=False)
            if cur_box is not None:
                box = self.add_box(cur_box, (1, 0, 0))
            else:
                box = pts.get_oriented_bounding_box()
                box.color = (1, 0, 0)
                self.vis.add_geometry(box, reset_bounding_box = False, waitKey=0)
            self.pre_geometries.append(pts)
            self.pre_geometries.append(box)
            
        if pre_box is not None:
            box2 = self.add_box(pre_box, (0, 0, 1))
            # self.pre_geometries.append(box)
        else:
            box2 = None
            
        while True:
            print(f'Is this {strs}? Press \'Y\' / \'N\'')
            state = self.vis.return_press_state()
            if state and file_path is not None:
                box.color = (0,1,0)
                self.vis.vis.update_geometry(box)
            
            elif file_path is not None:
                box.color = (0.5, 0.5, 0)
                self.vis.vis.update_geometry(box)
            
            if box2 is not None:
                box2.color = (0, 0.5, 0.5)
                self.vis.vis.update_geometry(box2)

            return state

    def get_box(self, frameid, humanid, tracking_results):
        # cur box postion
        frameid = int(frameid)
        ids = tracking_results[frameid]['ids']
        cur_pos = np.where(ids == int(humanid))[0][0]
        cur_box = tracking_results[frameid]['boxes_lidar'][cur_pos]
        return cur_box

    def is_too_far(self, frameid, humanid, tracking_results, framerate = 20):
        # cur box postion
        frameid = int(frameid)
        cur_box = self.get_box(frameid, humanid, tracking_results)

        # pre box postion
        pre_framid = int(self.checked_ids[humanid])
        pre_box = self.get_box(pre_framid, humanid, tracking_results)

        dist = np.linalg.norm(pre_box[:3] - cur_box[:3])
        vel = dist * framerate / (frameid - pre_framid)

        return abs(vel), pre_box, cur_box, pre_framid


    def choose_new_id(self, cur_box, cur_humanid, cur_framid, framerate=20):
        for humanid, box in self.pre_human_boxes.items():
            dist = np.linalg.norm(cur_box[:3] - box['box'][:3])
            duration = abs(int(cur_framid) - int(box['frameid']))
            vel = dist * framerate / duration
            if vel < 5 and duration<framerate:
                while humanid in self.reID:
                    humanid = self.reID[humanid]
                self.reID[cur_humanid] = humanid

    def filtering_method(self, frameid, humanid, tracking_folder, tracking_results, scene_path, filtered = False):

        file_path = self.join.join([tracking_folder, f'{humanid}_{frameid}.pcd'])
        scene_folder = self.join.join([os.path.dirname(tracking_folder), 'human_semantic'])
        # scene_path = list_dir_remote([self.load_data.client, scene_folder])[int(frameid)]
        scene_path = self.join.join([scene_folder, scene_path])

        is_person = False

        for geometry in self.pre_geometries:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)

        if humanid in self.checked_ids:
            
            vel, pre_box, cur_box, pre_framid = self.is_too_far(frameid, humanid, tracking_results)

            if vel > 5:
                print(f'Checking Human:{humanid} Cur frame:{frameid} (red) | Pre frame {pre_framid} (blue)')

                if filtered or self.interactive_choose(file_path=file_path, 
                                            scene_path=scene_path, 
                                            pre_box=pre_box, 
                                            cur_box=cur_box, 
                                            strs='a real human'):
                    is_person = True
                    # self.choose_new_id(cur_box, humanid, frameid)
                elif humanid in self.real_person_id:
                    # not a person, remove it from the real human list
                    self.real_person_id.pop(self.real_person_id.index(humanid))

            elif humanid in self.real_person_id:
                is_person = True

        else:
            print(f'Checking Human:{humanid} Frame:{frameid}')
            if filtered or self.interactive_choose(
                                file_path=file_path, 
                                scene_path=scene_path, 
                                strs='a real human'):
                cur_box = self.get_box(frameid, humanid, tracking_results)
                self.choose_new_id(cur_box, humanid, frameid)
                is_person = True


        return is_person

    def load_existing_tracking_list(self, tracking_folder):
        if self.remote:
            pcd_paths = list_dir_remote(self.load_data.client, tracking_folder)
        else:
            pcd_paths = os.listdir(tracking_folder)
        tracking_list = {}
        for pcd_path in pcd_paths:
            if not pcd_path.endswith('.pcd'):
                continue
            humanid = pcd_path.split('_')[0]
            frameid = pcd_path.split('_')[1].split('.')[0]
            if frameid in tracking_list:
                tracking_list[frameid].append(humanid)
            else:
                tracking_list[frameid] = [humanid, ]
        return tracking_list

    def run(self, filtered = False):
        tracking_list = self.load_existing_tracking_list(self.tracking_folder)
        basename = os.path.dirname(self.tracking_folder)

        if self.remote:
            tracking_file_path = f'{basename}/{os.path.basename(basename)}_tracking.pkl'
        else:
            tracking_file_path = self.join.join([basename, '0417-03_tracking.pkl'])
        tracking_results = self.load_data.load_pkl(tracking_file_path)

        scene_folder = self.join.join([os.path.dirname(self.tracking_folder), 'human_semantic'])
        scene_paths = self.load_data.list_dir(scene_folder)

        # with open(tracking_file_path, 'rb') as f:
        #     tracking_results = pkl.load(f)
        
        for frameid in sorted(tracking_list.keys()):
            for humanid in tracking_list[frameid]:
                if self.filtering_method(frameid, humanid, 
                                         self.tracking_folder, 
                                         tracking_results, 
                                         scene_paths[int(frameid)], filtered):
                    if humanid not in self.real_person_id:
                        self.real_person_id.append(humanid)
                    self.save_list.append(f'{humanid}_{frameid}.pcd')

                self.checked_ids[humanid] = frameid  # save previous frameid for humanid 
                
                cur_box = self.get_box(frameid, humanid, tracking_results)

                self.pre_human_boxes[humanid] = {'box': cur_box, 'frameid': frameid}
        
        self.copy_save_files()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=str,
                        help='A directory', default="C:\\Users\\DAI\\Desktop\\temp\\segment_by_tracking_03_slect")
    # parser.add_argument('--folder', '-f', type=str,
    #                     help='A directory', default="/hdd/dyd/lidarhumanscene/0417-03/segment_by_tracking")
    args, opts = parser.parse_known_args()
    # select_pcds_by_id(args.folder, ids)
    filter = filter_tracking_by_interactive(args.folder, remote=False)
    filter.run()

