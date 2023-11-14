import typing as t
import eagerx
import numpy as np
import gymnasium as gym
from collections import deque
import abc


TASK_STATUS = {"cancelled": -2, "pending": -1, "ongoing": 0, "success": 1}
GRIPPER_STATES = {"open": np.array([1.0], dtype="float32"), "closed": np.array([0.0], dtype="float32")}
RUNNING_STATES = {"running": 1, "stopping": 0}
REV_TASK_STATUS = {v: k for k, v in TASK_STATUS.items()}
REV_RUNNING_STATES = {v: k for k, v in RUNNING_STATES.items()}


# Define environment
class ArmEnv(eagerx.BaseEnv):
    def __init__(
        self,
        name,
        rate,
        graph,
        engine,
        backend,
        robot_type: str,
        render_mode: str = None,
        reset_ee_pose: t.List[float] = None,
        reset_gripper: t.List[float] = None,
        height: float = 0.12,
        force_start: bool = False,
        **kwargs,
    ):
        super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start, render_mode=render_mode)
        self._obs_space = self._observation_space
        self._act_space = self._action_space
        self._robot_type = robot_type
        self._reset_ee_pose = np.array(reset_ee_pose, dtype="float32")
        self._reset_gripper = np.array(reset_gripper, dtype="float32")
        self._height = height

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._obs_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._act_space

    def reset(self, seed: int = None, states: t.Optional[t.Dict[str, np.ndarray]] = None):
        # Set running_state & last_action
        self._steps = 0
        self._state = RUNNING_STATES["running"]
        self._action = self.action_space.sample()
        self._action.update(
            {
                "ee_pose": np.array(self._reset_ee_pose, dtype="float32"),
                "gripper": np.array(GRIPPER_STATES["open"], dtype="float32"),
            }
        )

        # Initialize task queue with a Wait task (wait for pick & place)
        self._task_queue: deque[Task] = deque([WaitForPickAndPlace("wait", height=self._height)])
        self._task_queue_done = []

        # Reset steps counter
        info = {}

        # Reset environment
        _states = self.state_space.sample()
        _defaul_reset_states = {
            "task/reset": np.array(1, dtype="int64"),  # Set to 0 if no task reset is needed
            f"{self._robot_type}/ee_pose": self._reset_ee_pose,
            f"{self._robot_type}/gripper": self._reset_gripper,
        }
        for k, v in _defaul_reset_states.items():
            if k in _states:
                _states[k] = v
        _states.update(states or {})
        obs = self._reset(_states)

        # Render
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _fake_observations(self, obs):
        if self._steps > 4:
            cmd = {
                "pick_pos": np.array([[0.3, -0.3, 0.15]], dtype="float32"),
                "pick_orn": np.array([[1, 0.0, 0, 0.0]], dtype="float32"),
                "place_pos": np.array([[0.3, 0.4, 0.15]], dtype="float32"),
                "place_orn": np.array([[1, 0.0, 0, 0.0]], dtype="float32"),
            }
            obs.update(cmd)
        else:
            cmd = {
                "pick_pos": np.array([[0, 0, 0]], dtype="float32"),
                "pick_orn": np.array([[0, 0, 0, 0]], dtype="float32"),
                "place_pos": np.array([[0, 0, 0]], dtype="float32"),
                "place_orn": np.array([[0, 0, 0, 0]], dtype="float32"),
            }
            obs.update(cmd)
        if 65 < self._steps < 90:
            obs["stop"] = np.array([True], dtype="bool")
        else:
            obs["stop"] = np.array([False], dtype="bool")
        return obs

    def step(self, _prev_obs):
        # Step the environment
        obs = self._step(self._action)
        self._steps += 1

        # Used for debugging
        # obs = self._fake_observations(obs)

        stopping = obs.get("stop", np.array([False], dtype="bool"))[-1]
        if stopping and self._state == RUNNING_STATES["running"]:
            self._task_queue.appendleft(Stop("stop"))
            self._state = RUNNING_STATES["stopping"]

        # Update task queue
        info = {}
        while len(self._task_queue) > 0:
            task = self._task_queue[0]
            status = task.update(obs)
            if status in [TASK_STATUS["cancelled"], TASK_STATUS["success"]]:
                self._task_queue_done.append(self._task_queue.popleft())
                print(f"Task `{task.name}` is `{REV_TASK_STATUS[status]}`.")
                task.done_callback(self, obs, self._task_queue)
            elif status == TASK_STATUS["pending"]:
                raise ValueError("Task status cannot be pending.")
            elif status == TASK_STATUS["ongoing"]:
                self._action = task.get_action(self._action)
                break

        # Terminate if task queue is empty
        terminated = len(self._task_queue) == 0
        truncated = False
        rwd = 0.0

        # Render
        if self.render_mode == "human":
            self.render()
        return obs, rwd, terminated, truncated, info


class Task:
    def __init__(self, name: str, status: int = TASK_STATUS["pending"]):
        self._name = name
        self._status = status
        self._partial_action: t.Dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> int:
        return self._status

    def set_partial_action(self, partial_action: t.Dict[str, np.ndarray]):
        self._partial_action = partial_action

    def set_status(self, status: int):
        self._status = status

    def get_action(self, base_action: t.Dict[str, np.ndarray]) -> t.Dict[str, np.ndarray]:
        action = {k: self._partial_action.get(k, v) for k, v in base_action.items()}
        return action

    def update(self, obs: t.Dict[str, np.ndarray]) -> int:
        # Check if task is ongoing
        if self._status == TASK_STATUS["pending"]:
            self.set_status(TASK_STATUS["ongoing"])
        elif self._status == TASK_STATUS["ongoing"]:
            pass
        elif self._status in [TASK_STATUS["success"], TASK_STATUS["cancelled"]]:
            return self._status
        else:
            raise ValueError("Invalid task status.")
        self._update(obs)
        return self.status

    def done_callback(self, env: ArmEnv, obs: t.Dict[str, np.ndarray], queue: deque) -> None:
        assert self.status in [TASK_STATUS["success"], TASK_STATUS["cancelled"]]
        self._done_callback(env, obs, queue)

    @abc.abstractmethod
    def _update(self, obs: t.Dict[str, np.ndarray]) -> None:
        pass

    def _done_callback(self, env: ArmEnv, obs: t.Dict[str, np.ndarray], queue: deque) -> None:
        pass


class WaitForPickAndPlace(Task):
    def __init__(self, name: str, height: float = 0.1):
        super().__init__(name)
        self._height = height
        self._wait_cmd = {
            "pick_pos": np.array([0, 0, 0], dtype="float32"),
            "pick_orn": np.array([0, 0, 0, 0], dtype="float32"),
            "place_pos": np.array([0, 0, 0], dtype="float32"),
            "place_orn": np.array([0, 0, 0, 0], dtype="float32"),
        }

    def _update(self, obs: t.Dict[str, np.ndarray]):
        # Check if pick & place have changed.
        has_cmd = False
        for k, v in self._wait_cmd.items():
            if k in obs and not np.allclose(obs[k][-1], v):
                has_cmd = True
            else:
                has_cmd = False
                break

        if has_cmd:
            self.set_status(TASK_STATUS["success"])

    def _done_callback(self, env: ArmEnv, obs: t.Dict[str, np.ndarray], queue: deque):
        if self.status == TASK_STATUS["success"]:
            # Add pick task
            pick_task = Pick("pick", obs["pick_pos"][-1], obs["pick_orn"][-1], height=self._height)
            # Add place task
            place_task = Place("place", obs["place_pos"][-1], obs["place_orn"][-1], height=self._height)
            # Add tasks to queue
            # Notice reverse order, because we are left-extending the queue.
            queue.extendleft(reversed([pick_task, place_task]))


class Stop(Task):
    # todo: Also stop gripper at current position
    def __init__(self, name: str):
        super().__init__(name)
        self._has_cmd = False

    def _update(self, obs: t.Dict[str, np.ndarray]):
        # Determine if first time entering this task
        if not self._has_cmd:
            ee_pose = np.concatenate([obs["ee_pos"][-1], obs["ee_orn"][-1]])
            action_stop = {"ee_pose": ee_pose}
            self.set_partial_action(action_stop)
            self._has_cmd = True

        # Check if still need to stop
        stopping = obs.get("stop", np.array([False], dtype="bool"))
        if not stopping:
            self.set_status(TASK_STATUS["success"])

    def _done_callback(self, env: ArmEnv, obs: t.Dict[str, np.ndarray], queue: deque):
        if self.status == TASK_STATUS["success"]:
            env._state = RUNNING_STATES["running"]


class MoveEE(Task):
    def __init__(self, name: str, ee_pos: np.ndarray, ee_orn: np.ndarray, tol: float = 0.01, num_updates: int = 100):
        super().__init__(name)
        self._ee_pos = ee_pos
        self._ee_orn = ee_orn
        self._ee_pose = np.concatenate([self._ee_pos, self._ee_orn])
        self._tol = tol
        self._num_updates = num_updates
        self._updates = 0

    def _update(self, obs: t.Dict[str, np.ndarray]):
        # Check if ee_pos, ee_orn are reached within tolerance
        ee_pose = np.concatenate([obs["ee_pos"][-1], obs["ee_orn"][-1]])
        is_done = np.isclose(ee_pose, self._ee_pose, atol=self._tol).all()
        is_timeout = self._updates > self._num_updates

        # Update task status
        if is_done:
            self._updates = 0
            self.set_status(TASK_STATUS["success"])
        elif is_timeout:
            self._updates = 0
            self.set_status(TASK_STATUS["cancelled"])
        else:
            self._updates += 1
        # Update partial action
        partial_action = {"ee_pose": self._ee_pose}
        self.set_partial_action(partial_action)


class MoveGripper(Task):
    def __init__(self, name: str, gripper: str = "open", num_updates: int = 5, tol=5e-3):
        super().__init__(name)
        self._gripper = GRIPPER_STATES[gripper]
        self._updates = 0
        self._tol = tol
        self._num_updates = num_updates
        self._last_gripper_pos = None

    def _update(self, obs: t.Dict[str, np.ndarray]):
        curr = obs.get("gripper_pos")[-1][0]
        self._last_gripper_pos = curr if self._last_gripper_pos is None else self._last_gripper_pos

        if abs(self._last_gripper_pos - curr) < self._tol:
            self._updates += 1
        else:
            # Gripper is moving
            self._last_gripper_pos = curr
            self._updates = 0
        is_done = self._updates > self._num_updates

        # Update task status
        if is_done:
            self.set_status(TASK_STATUS["success"])

        # Update partial action
        partial_action = {"gripper": self._gripper}
        self.set_partial_action(partial_action)


class Pick(Task):
    def __init__(self, name: str, ee_pos: np.ndarray, ee_orn: np.ndarray, tol: float = 0.01, height: float = 0.1):
        super().__init__(name)
        self._ee_pos = ee_pos
        self._ee_orn = ee_orn
        self._height = height
        self._tol = tol

    def _update(self, obs: t.Dict[str, np.ndarray]) -> None:
        # Set status to success to schedule next tasks
        self.set_status(TASK_STATUS["success"])

    def _done_callback(self, env: ArmEnv, obs: t.Dict[str, np.ndarray], queue: deque):
        if self.status == TASK_STATUS["success"]:
            # todo: wrap update with function (and not _update!!) that checks if pose has not changed. If so, cancel.
            # Pre-grasp pose
            ee_pos_pre = np.array([self._ee_pos[0], self._ee_pos[1], self._ee_pos[2] + self._height], dtype="float32")
            ee_orn_pre = self._ee_orn
            task_pre = MoveEE(f"{self.name}/pre_grasp to xyz={ee_pos_pre}", ee_pos_pre, ee_orn_pre, tol=self._tol)
            task_open = MoveGripper(f"{self.name}/open", gripper="open", num_updates=10, tol=5e-3)
            # Grasp pose
            task_grasp = MoveEE(f"{self.name}/grasp at xyz={self._ee_pos}", self._ee_pos, self._ee_orn, tol=self._tol)
            task_close = MoveGripper(f"{self.name}/close", gripper="closed", num_updates=15, tol=5e-3)
            # Grasp
            task_lift = MoveEE(f"{self.name}/post_grasp to xyz={ee_pos_pre}", ee_pos_pre, ee_orn_pre, tol=self._tol)
            # Add tasks to queue
            # Notice reverse order, because we are left-extending the queue.
            queue.extendleft(reversed([task_pre, task_open, task_grasp, task_close, task_lift]))


class Place(Task):
    def __init__(self, name: str, ee_pos: np.ndarray, ee_orn: np.ndarray, tol: float = 0.01, height: float = 0.1):
        super().__init__(name)
        self._ee_pos = ee_pos
        self._ee_orn = ee_orn
        self._height = height
        self._tol = tol

    def _update(self, obs: t.Dict[str, np.ndarray]) -> None:
        # Set status to success to schedule next tasks
        self.set_status(TASK_STATUS["success"])

    def _done_callback(self, env: ArmEnv, obs: t.Dict[str, np.ndarray], queue: deque):
        if self.status == TASK_STATUS["success"]:
            # Pre-place pose
            ee_pos_pre = np.array([self._ee_pos[0], self._ee_pos[1], self._ee_pos[2] + self._height], dtype="float32")
            ee_orn_pre = self._ee_orn
            task_pre = MoveEE(f"{self.name}/pre_place to xyz={ee_pos_pre}", ee_pos_pre, ee_orn_pre, tol=self._tol)
            # Place pose
            task_place = MoveEE(
                f"{self.name}/place at xyz={self._ee_pos}", self._ee_pos, self._ee_orn, tol=self._tol
            )  # TODO: INCREASE TOLERANCE & NUM_UPDATES DUE TO SPIRALING SEARCH.
            task_open = MoveGripper(f"{self.name}/open", gripper="open", num_updates=10, tol=5e-3)
            # Return to pre-place pose
            task_lift = MoveEE(f"{self.name}/post_place to xyz={ee_pos_pre}", ee_pos_pre, ee_orn_pre, tol=self._tol)
            # Add tasks to queue
            # Notice reverse order, because we are left-extending the queue.
            queue.extendleft(reversed([task_pre, task_place, task_open, task_lift]))
