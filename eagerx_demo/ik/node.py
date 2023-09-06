from typing import Optional, List, Dict
from collections import deque
import numpy as np
from dataclasses import dataclass

import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg, load

# ROS imports
from urdf_parser_py.urdf import URDF


@dataclass
class IndexedJointObject:
    """Index of a robot joint and its name."""

    joint_name: str
    joint_uid: int


def index_joints_and_ee_link(pb, physics_uid, robot_id, joints, ee_link):
    """Map a list of joint names to indexed joints.
    In other words, map named joints to the index used by
    PyBullet to facilitate setting the configuration.
    Parameters:
      physics_uid: Index of the PyBullet physics server to use.
      joints: list with joint name keys
      ee_link: end effector link name
    Returns: a list of IndexedJointObject
    """
    indexed_joints = []
    indexed_ee_link = None
    n = pb.getNumJoints(robot_id, physics_uid)
    for joint_name in joints:
        for i in range(n):
            info = pb.getJointInfo(robot_id, i, physics_uid)
            if info[12].decode("utf8") == ee_link:
                indexed_ee_link = i
            if joint_name == info[1].decode("utf-8"):
                indexed_joints.append(IndexedJointObject(joint_name, i))
                continue
    assert len(joints) == len(indexed_joints), "Not all joints were found in the provided urdf."
    assert indexed_ee_link is not None, "End effector link not found in the provided urdf."
    return indexed_joints, indexed_ee_link


class TaskSpaceControl(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: List[int],
        upper: List[float],
        lower: List[float],
        ee_link: str,
        rest_poses: List[float],
        robot_dict: Dict,
        gui: bool = False,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> NodeSpec:
        """
        Filters goal joint positions that cause self-collisions or are below a certain height.
        Also check velocity limits.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param joints: joint names
        :param upper: upper joint limits
        :param lower: lower joint limits
        :param ee_link: end effector link name
        :param rest_poses: rest poses of the robot
        :param robot_dict: robot dict
        :param gui: whether to use gui
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["ee_pos", "ee_orn", "position"]
        spec.config.outputs = ["goal"]

        # Modify custom node params
        spec.config.joints = joints
        spec.config.upper = upper
        spec.config.lower = lower
        spec.config.ee_link = ee_link
        spec.config.rest_poses = rest_poses
        spec.config.gui = gui

        # Collision detector
        assert isinstance(robot_dict, dict), "robot_dict must be a dict"
        assert "urdf" in robot_dict, "robot_dict must contain a urdf key"
        assert "basePosition" in robot_dict, "robot_dict must contain a basePosition key"
        assert "baseOrientation" in robot_dict, "robot_dict must contain a baseOrientation key"
        spec.config.robot_dict = robot_dict if isinstance(robot_dict, dict) else None

        # Add converter & space
        spec.inputs.position.space.update(low=lower, high=upper)
        spec.outputs.goal.space.update(low=lower, high=upper)
        return spec

    def initialize(self, spec: NodeSpec):
        self.joints = spec.config.joints
        self.upper = np.array(spec.config.upper, dtype="float")
        self.lower = np.array(spec.config.lower, dtype="float")
        self.ee_link = spec.config.ee_link
        self.rest_poses = np.array(spec.config.rest_poses, dtype="float")
        self.robot_dict = spec.config.robot_dict
        self.gui = spec.config.gui

        # Setup physics server for ik solver
        import pybullet as pb

        self.pb = pb
        if self.gui:
            self._p = pb.connect(pb.GUI)
        else:
            self._p = pb.connect(pb.DIRECT)
        # Load workspace
        # bodies = load(self.collision["workspace"])(self._p)
        self.arm = {}
        # Generate robot urdf (if not a path but a text file)
        r = self.robot_dict
        if r["urdf"].endswith(".urdf"):  # Full path specified
            fileName = r["urdf"]
        else:  # First write to /tmp file (else pybullet cannot load urdf)
            import uuid  # Use this to generate a unique filename

            fileName = f"/tmp/{str(uuid.uuid4())}.urdf"
            with open(fileName, "w") as file:
                file.write(r["urdf"])
        # Load robot
        self.arm = pb.loadURDF(
            fileName,
            basePosition=r.get("basePosition", None),
            baseOrientation=r.get("baseOrientation", None),
            useFixedBase=r.get("useFixedBase", True),
            flags=r.get("flags", 0),
            physicsClientId=self._p,
        )
        self.indexed_joints, self.index_ee_link = index_joints_and_ee_link(pb, self._p, self.arm, self.joints, self.ee_link)

    @register.states()
    def reset(self):
        self._last_ee_pose_goal = None
        self._last_goal = None

    @register.inputs(
        ee_pos=Space(low=[-2, -2, 0], high=[2, 2, 2], dtype="float32"),
        ee_orn=Space(low=-1, high=1, shape=(4,), dtype="float32"),
        position=Space(dtype="float32"),
    )
    @register.outputs(goal=Space(dtype="float32"))
    def callback(self, t_n: float, ee_pos: Msg = None, ee_orn: Msg = None, position: Msg = None):
        ee_pos_goal = ee_pos.msgs[-1]
        ee_orn_goal = ee_orn.msgs[-1]
        ee_pose_goal = np.concatenate([ee_pos_goal, ee_orn_goal])

        # Set to current position
        current = position.msgs[-1]
        for i, joint in enumerate(self.indexed_joints):
            self.pb.resetJointState(self.arm, joint.joint_uid, current[i], physicsClientId=self._p)

        # Determine status: 0: ongoing, 1: success
        if self._last_ee_pose_goal is None or not np.allclose(self._last_ee_pose_goal, ee_pose_goal):
            self._last_ee_pose_goal = ee_pose_goal
            run_ik = True
        else:
            run_ik = False

        if not run_ik and self._last_goal is not None:
            return dict(goal=self._last_goal.astype("float32"))
        else:
            goal = self.pb.calculateInverseKinematics(
                bodyUniqueId=self.arm,
                endEffectorLinkIndex=self.index_ee_link,
                targetPosition=ee_pos_goal,
                targetOrientation=ee_orn_goal,
                lowerLimits=self.lower,
                upperLimits=self.upper,
                jointRanges=self.upper - self.lower,
                restPoses=self.rest_poses,
                # currentPositions=current.tolist(),
                maxNumIterations=100,
                residualThreshold=1e-5,
                physicsClientId=self._p,
            )
            goal = np.array(goal[0:6], dtype="float32")
            # goal[2:] = (goal[2:] + np.pi) % (2 * np.pi) - np.pi
            self._last_goal = goal
            return dict(goal=goal)
