from scipy.spatial.transform import Rotation as R
import typing as t
import eagerx
import numpy as np
import gymnasium as gym


# Define environment
class ArmEnv(eagerx.BaseEnv):
    def __init__(self, name, rate, graph, engine, backend, max_steps: int, render_mode: str = None):
        super().__init__(name, rate, graph, engine, backend=backend, force_start=False, render_mode=render_mode)
        self.steps = 0
        self.max_steps = max_steps

        # Exclude
        self._exclude_list = ["pos", "pos_desired"]

    @property
    def observation_space(self) -> gym.spaces.Space:
        obs_space = self._observation_space
        names = obs_space.keys()
        for o in self._exclude_list:
            if o in names:
                low, high = obs_space[o].low[:, :2], obs_space[o].high[:, :2]
                obs_space[o] = gym.spaces.Box(low=low, high=high, dtype="float32")
        return obs_space

    def _exclude_obs(self, obs):
        names = obs.keys()
        for o in self._exclude_list:
            if o in names:
                obs[o] = obs[o][:, :2]
        return obs

    def step(self, action):
        # Step the environment
        self.steps += 1
        info = {}
        obs = self._step(action)

        # Calculate reward
        ee_pos = obs["ee_position"][0]
        rwd = 0.

        truncated = self.steps >= self.max_steps
        terminated = False

        # Exclude
        obs = self._exclude_obs(obs)

        # Render
        if self.render_mode == "human":
            self.render()
        return obs, rwd, terminated, truncated, info

    def reset(self, seed: int = None, states: t.Optional[t.Dict[str, np.ndarray]] = None):
        # Reset steps counter
        self.steps = 0
        info = {}

        # Reset environment
        _states = self.state_space.sample()
        obs = self._reset(_states)

        # Exclude
        obs = self._exclude_obs(obs)

        # Render
        if self.render_mode == "human":
            self.render()
        return obs, info
