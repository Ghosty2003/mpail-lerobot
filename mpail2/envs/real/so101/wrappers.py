"""MPAIL-shaped wrapper for SO101RobotEnv (mirrors FrankaRealWrapper / KinovaRealWrapper)."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from .robot_limits import MAX_EPISODE_STEPS


class SO101RealWrapper(gym.Wrapper):
    """Converts SO101RobotEnv numpy observations to torch tensors.

    Sets ``num_envs`` and ``max_episode_length`` on the *inner* env so that
    ``env.unwrapped`` exposes both attributes as required by MPAIL2Runner.
    """

    def __init__(self, env: gym.Env, device: str = "cuda"):
        super().__init__(env)
        self.device = device
        self.step_count = 0
        self.episode_count = 0

        # Expose these on unwrapped for MPAIL2Runner
        self.env.unwrapped.num_envs = 1
        if not hasattr(self.env.unwrapped, "max_episode_length"):
            self.env.unwrapped.max_episode_length = MAX_EPISODE_STEPS
        self.max_episode_length = self.env.unwrapped.max_episode_length

    # ─── observation helpers ──────────────────────────────────────────────────

    def _obs_to_torch(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in obs.items()
        }

    # ─── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self.step_count = 0
        self.episode_count += 1
        t0 = time.time()
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["reset_time"] = info.get("reset_time", round(time.time() - t0, 3))
        return self._obs_to_torch(obs), info

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        t0 = time.time()

        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action_np)

        self.step_count += 1
        truncated = truncated or (self.step_count >= self.max_episode_length)

        info = dict(info)
        info["action_executed"] = action_np.flatten().tolist()
        info["mpail_env/step_time"] = round(time.time() - t0, 3)

        return (
            self._obs_to_torch(obs),
            torch.tensor([float(reward)], dtype=torch.float32, device=self.device),
            torch.tensor([bool(terminated)], dtype=torch.bool, device=self.device),
            torch.tensor([bool(truncated)], dtype=torch.bool, device=self.device),
            info,
        )

    def seed(self, seed: int | None = None) -> None:
        pass
