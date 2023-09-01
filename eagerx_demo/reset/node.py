from typing import List
import numpy as np
import eagerx
from eagerx import Space
from eagerx.core.specs import ResetNodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg


class ResetArm(eagerx.ResetNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        upper: List[float],
        lower: List[float],
        gripper: bool = True,
        threshold: float = 0.02,
        timeout: float = 4.0,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> ResetNodeSpec:
        """Resets joints & Gripper to goal_joints positions.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param upper: Upper joint limits
        :param lower: Lower joint limits
        :param gripper: Include a gripper
        :param threshold: Closeness to the goal_joints before considering the reset to be finished.
        :param timeout: Seconds before considering the reset to be finished, regardless of the closeness. A value of `0` means
                        indefinite.
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
        spec.config.inputs = ["joints"]
        spec.config.targets = ["goal_joints"]
        spec.config.outputs = ["joints"]
        # Add gripper of also controlled
        if gripper:
            spec.config.outputs.append("gripper")
            spec.config.targets.append("goal_gripper")
        # Add custom params
        spec.config.threshold = threshold
        spec.config.timeout = timeout

        # Add variable space
        spec.inputs.joints.space.update(low=lower, high=upper)
        spec.targets.goal_joints.space.update(low=lower, high=upper)
        spec.outputs.joints.space.update(low=lower, high=upper)
        return spec

    def initialize(self, spec: ResetNodeSpec):
        self.threshold = spec.config.threshold
        self.timeout = spec.config.timeout

    @register.states()
    def reset(self):
        self.start = None

    @register.inputs(joints=Space(dtype="float32"))
    @register.targets(goal_joints=Space(dtype="float32"), goal_gripper=Space(low=[0.0], high=[1.0]))
    @register.outputs(joints=Space(dtype="float32"), gripper=Space(low=[0.0], high=[1.0]))
    def callback(self, t_n: float, goal_joints: Msg = None, goal_gripper: Msg = None, joints: Msg = None):
        if self.start is None:
            self.start = t_n

        # Process goal_joints & current joint msgs
        joints = joints.msgs[-1]
        goal_joints = goal_joints.msgs[-1]
        goal_gripper = goal_gripper.msgs[-1]

        # Determine done flag
        duration = (t_n - self.start)
        if duration > 1.0 and np.isclose(joints, goal_joints, atol=self.threshold).all():
            is_done = True
        else:
            if self.timeout > 0 and self.timeout < duration:
                is_done = True
            else:
                is_done = False

        # Create output message
        output_msgs = dict(joints=goal_joints, gripper=goal_gripper)
        output_msgs["goal_joints/done"] = is_done
        output_msgs["goal_gripper/done"] = is_done
        return output_msgs

