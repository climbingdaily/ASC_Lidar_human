from scipy.spatial.transform import Rotation as R, rotation

import math
import numpy as np
import pandas as pd

import os
import sys
import torch
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_PATH)

from util import transformation
from smpl import model


class MoCapData():
    def __init__(self, worldpos_csv, rotation_csv):
        self.worldpos_df = pd.read_csv(worldpos_csv)
        self.rotation_df = pd.read_csv(rotation_csv)

        self.smpl_to_mocap = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg',
                              'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2',
                              'LeftFootEnd', 'RightFootEnd', 'Neck', 'LeftShoulder',
                              'RightShoulder', 'Head', 'LeftArm', 'RightArm',
                              'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
                              'LeftHandThumb2', 'RightHandThumb2']

        self.trans = np.array(
            [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=np.float32)
        assert len(self.worldpos_df) == len(self.rotation_df)

    def __len__(self):
        return len(self.worldpos_df)

    def worldpos(self, index):
        return transformation.rotation(np.array(self.worldpos_df.iloc[[index]].values.tolist()[0][1:]).reshape(-1, 3) / 100, self.trans)

    def pose(self, index):

        pose = []
        import pandas as pd
        for each in self.smpl_to_mocap:
            xrot = math.radians(self.rotation_df.at[index, each + '.X'])
            yrot = math.radians(self.rotation_df.at[index, each + '.Y'])
            zrot = math.radians(self.rotation_df.at[index, each + '.Z'])

            # zrot offset provided by shtu
            if each == 'LeftShoulder':
                zrot += -0.3
            elif each == 'RightShoulder':
                zrot += 0.3
            elif each == 'LeftArm':
                zrot += 0.3
            elif each == 'RightArm':
                zrot += -0.3

            pose.append(R.from_euler('zxy', [zrot, xrot, yrot]).as_rotvec())
        pose = np.stack(pose).flatten()
        return pose

    def translation(self, index, untranslated_root=None):
        if untranslated_root is None:
            untranslated_root = self.smpl_joints_untranslated(index)[0]
        return self.worldpos(index)[0] - untranslated_root

    def smpl_vertices(self, index, beta=None, is_torch=False):
        vertices = model.get_vertices(
            self.pose(index), beta, return_numpy=not is_torch)
        trans = self.trans
        translation = self.translation(index)
        if is_torch:
            trans = torch.from_numpy(trans)
            translation = torch.from_numpy(translation).to(vertices.device)
        return transformation.rotation(vertices, trans) + translation

    def smpl_joints_untranslated(self, index):
        return transformation.rotation(model.get_vertices(pose=self.pose(index), return_joints=True), self.trans)

    def smpl_joints(self, index):
        smpl_joints_untranslated = self.smpl_joints_untranslated(index)
        return smpl_joints_untranslated + self.translation(index, smpl_joints_untranslated[0])
