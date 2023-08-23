"""Aligning task."""

import os

import numpy as np
from eagerx_demo.cliport.tasks.task import Task
from eagerx_demo.cliport.utils import utils


class AlignBoxCorner(Task):
    """Aligning task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "align the brown box with the green corner"
        self.task_completed_desc = "done with alignment"

    def reset(self, env):
        super().reset(env)

        # Generate randomly shaped box.
        box_size = self.get_random_size(0.05, 0.15, 0.05, 0.15, 0.01, 0.06)

        # Add corner.
        dimx = (box_size[0] / 2 - 0.025 + 0.0025, box_size[0] / 2 + 0.0025)
        dimy = (box_size[1] / 2 + 0.0025, box_size[1] / 2 - 0.025 + 0.0025)
        corner_template = 'corner/corner-template.urdf'
        replace = {'DIMX': dimx, 'DIMY': dimy}
        corner_urdf = self.fill_template(corner_template, replace)
        corner_size = (box_size[0], box_size[1], 0)
        corner_pose = self.get_random_pose(env, corner_size)
        env.add_object(corner_urdf, corner_pose, 'fixed')
        if os.path.exists(corner_urdf):
            os.remove(corner_urdf)

        # Add possible placing poses.
        theta = utils.quatXYZW_to_eulerXYZ(corner_pose[1])[2]
        fip_rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta + np.pi))
        pose1 = (corner_pose[0], fip_rot)
        alt_x = (box_size[0] / 2) - (box_size[1] / 2)
        alt_y = (box_size[1] / 2) - (box_size[0] / 2)
        alt_pos = (alt_x, alt_y, 0)
        alt_rot0 = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
        alt_rot1 = utils.eulerXYZ_to_quatXYZW((0, 0, 3 * np.pi / 2))
        pose2 = utils.multiply(corner_pose, (alt_pos, alt_rot0))
        pose3 = utils.multiply(corner_pose, (alt_pos, alt_rot1))

        # Add box.
        box_template = 'box/box-template.urdf'
        box_urdf = self.fill_template(box_template, {'DIM': box_size})
        box_pose = self.get_random_pose(env, box_size)
        box_id = env.add_object(box_urdf, box_pose)
        if os.path.exists(box_urdf):
            os.remove(box_urdf)
        self.color_random_brown(box_id)

        # Goal: box is aligned with corner (1 of 4 possible poses).
        self.goals.append(([(box_id, (2 * np.pi, None))], np.int32([[1, 1, 1, 1]]),
                           [corner_pose, pose1, pose2, pose3],
                           False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)
