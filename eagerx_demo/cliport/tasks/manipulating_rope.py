"""Cable task."""

import os

import numpy as np
from eagerx_demo.cliport.tasks import primitives
from eagerx_demo.cliport.tasks.task import Task
from eagerx_demo.cliport.utils import utils

import pybullet as p


class ManipulatingRope(Task):
    """Cable task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.pos_eps = 0.02
        self.primitive = primitives.PickPlace(height=0.02, speed=0.001)
        self.lang_template = "manipulate the rope to complete the square"
        self.task_completed_desc = "done manipulating the rope."

    def reset(self, env):
        super().reset(env)

        n_parts = 20
        radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)

        # Add 3-sided square.
        square_size = (length, length, 0)
        square_pose = self.get_random_pose(env, square_size)
        square_template = "square/square-template.urdf"
        replace = {"DIM": (length,), "HALF": (length / 2 - 0.005,)}
        urdf = self.fill_template(square_template, replace)
        env.add_object(urdf, square_pose, "fixed")
        if os.path.exists(urdf):
            os.remove(urdf)

        # Get corner points of square.
        corner0 = (length / 2, length / 2, 0.001)
        corner1 = (-length / 2, length / 2, 0.001)
        corner0 = utils.apply(square_pose, corner0)
        corner1 = utils.apply(square_pose, corner1)

        # Add cable (series of articulated small blocks).
        increment = (np.float32(corner1) - np.float32(corner0)) / n_parts
        position, _ = self.get_random_pose(env, (0.1, 0.1, 0.1))
        position = np.float32(position)
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
        parent_id = -1
        targets = []
        objects = []
        for i in range(n_parts):
            position[2] += np.linalg.norm(increment)
            part_id = p.createMultiBody(0.1, part_shape, part_visual, basePosition=position)
            if parent_id > -1:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=parent_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, np.linalg.norm(increment)),
                    childFramePosition=(0, 0, 0),
                )
                p.changeConstraint(constraint_id, maxForce=100)
            if (i > 0) and (i < n_parts - 1):
                color = utils.COLORS["red"] + [1]
                p.changeVisualShape(part_id, -1, rgbaColor=color)
            env.obj_ids["rigid"].append(part_id)
            parent_id = part_id
            target_xyz = np.float32(corner0) + i * increment + increment / 2
            objects.append((part_id, (0, None)))
            targets.append((target_xyz, (0, 0, 0, 1)))

        matches = np.clip(np.eye(n_parts) + np.eye(n_parts)[::-1], 0, 1)

        self.goals.append((objects, matches, targets, False, False, "pose", None, 1))
        self.lang_goals.append(self.lang_template)

        for i in range(480):
            p.stepSimulation()
