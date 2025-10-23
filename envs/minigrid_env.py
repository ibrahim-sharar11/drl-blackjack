import gymnasium as gym
from gymnasium import spaces
import numpy as np
import minigrid  # ensure IDs registered

class MiniGridWrapper(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, env_id="MiniGrid-Empty-8x8-v0", max_steps=200, reward_weights=None, seed=7, obs_mode="partial", reward_scale=1.0, render_mode=None):
        super().__init__()
        self.rw = reward_weights or {}
        self.reward_scale = reward_scale
        if not any(env_id.endswith(suf) for suf in ("-v0","-v1","-v2")):
            env_id = f"{env_id}-v0"
        # Use headless mode for training/CI by default; allow rgb_array for recording
        self.base_env = gym.make(env_id, max_steps=max_steps, render_mode=render_mode)
        self.base_env.reset(seed=seed)
        obs, _ = self.base_env.reset()
        img = obs["image"].astype(np.float32) / 10.0
        self.observation_space = spaces.Box(low=0, high=1, shape=img.shape, dtype=np.float32)
        self.action_space = self.base_env.action_space
        self.max_steps = max_steps
        self.steps = 0
        self.visited = set()
        self.success = 0

    def _norm(self, obs):
        return obs["image"].astype(np.float32) / 10.0

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed)
        self.steps = 0
        # Track true agent positions for coverage (less collision-prone)
        pos = self.base_env.unwrapped.agent_pos
        agent_pos = tuple(pos if isinstance(pos, (list, tuple)) else pos.tolist())
        self.visited = {agent_pos}
        self.success = 0
        return self._norm(obs), info or {}

    def step(self, action):
        obs, r, terminated, truncated, info = self.base_env.step(action)
        self.steps += 1
        shaped = self.rw.get("step_cost", 0.0)
        if terminated and r > 0:
            self.success = 1
            shaped += self.rw.get("success", 1.0)
        # Update coverage by agent grid positions
        pos = self.base_env.unwrapped.agent_pos
        agent_pos = tuple(pos if isinstance(pos, (list, tuple)) else pos.tolist())
        if agent_pos not in self.visited:
            self.visited.add(agent_pos)
            shaped += self.rw.get("explore_bonus", 0.0)
        if terminated:
            shaped += self.rw.get("speed_bonus", 0.0) * (self.max_steps - self.steps) / self.max_steps
        metrics = {"steps": self.steps, "visited_cells": len(self.visited), "success": self.success}
        return self._norm(obs), float(r) * self.reward_scale + shaped, terminated, truncated, metrics
