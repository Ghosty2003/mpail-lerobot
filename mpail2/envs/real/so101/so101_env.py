"""SO-101 (SO-ARM101) base Gymnasium environment.

Communicates with a lightweight HTTP robot-control server running in the
LeRobot conda environment.  The server must expose three endpoints:

    GET  /state
        Response: {"joints": [float x 6]}
        Returns current joint positions in degrees, in JOINT_NAMES order.

    POST /step
        Body:     {"joints": [float x 6]}
        Execute a joint-position command and return immediately.
        Response: {"ok": true}

    POST /reset
        Move the arm to HOME_POSITION_DEG and open the gripper.
        Response: {"ok": true}

Minimal FastAPI server skeleton for the LeRobot side::

    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()
    robot = ...  # your lerobot SO101 instance

    @app.get("/state")
    def state():
        pos = robot.get_joint_positions()   # list of 6 floats (degrees)
        return {"joints": pos}

    @app.post("/step")
    def step(body: dict):
        robot.send_joint_positions(body["joints"])
        return {"ok": True}

    @app.post("/reset")
    def reset():
        robot.go_to_home()
        return {"ok": True}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=7070)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .robot_limits import (
    ACTION_DIM,
    CAM_C,
    CAM_H,
    CAM_KEY,
    CAM_W,
    CONTROL_FREQUENCY_HZ,
    DEFAULT_ROBOT_HOST,
    DEFAULT_ROBOT_PORT,
    HOME_POSITION_DEG,
    JOINT_LOWER_DEG,
    JOINT_NAMES,
    JOINT_UPPER_DEG,
    MAX_DELTA_DEG,
    MAX_EPISODE_STEPS,
    STATE_DIM,
)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    print("[WARNING] `requests` not installed — SO101RobotEnv will run in mock mode.")


# obs key used by planner_server.py (--obs_key default)
OBS_KEY = "observation.state"


class SO101RobotEnv(gym.Env):
    """Gymnasium env that wraps the SO-101 arm via an HTTP robot-control server.

    ``observation_space`` and ``action_space`` use a leading batch dimension
    of 1 so shapes match MPAIL2Runner expectations directly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = DEFAULT_ROBOT_HOST,
        port: int = DEFAULT_ROBOT_PORT,
        control_frequency: float = CONTROL_FREQUENCY_HZ,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        mock: bool = False,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.control_frequency = control_frequency
        self._step_dt = 1.0 / control_frequency
        self.max_episode_steps = max_episode_steps
        self.mock = mock or not _REQUESTS_AVAILABLE

        # runner.py reads these from env.unwrapped
        self.num_envs = 1

        self.observation_space = spaces.Dict({
            OBS_KEY: spaces.Box(
                low=np.tile(JOINT_LOWER_DEG, (1, 1)),
                high=np.tile(JOINT_UPPER_DEG, (1, 1)),
                shape=(1, STATE_DIM),
                dtype=np.float32,
            ),
            CAM_KEY: spaces.Box(
                low=0.0, high=1.0,
                shape=(1, CAM_H, CAM_W, CAM_C),
                dtype=np.float32,
            ),
        })
        # Actions are normalised joint targets in [-1, 1] (planner outputs tanh)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, ACTION_DIM), dtype=np.float32
        )

        self._current_joints = HOME_POSITION_DEG.copy()

        if self.mock:
            print("[SO101RobotEnv] Running in mock mode — no HTTP calls will be made.")
        else:
            print(f"[SO101RobotEnv] Connecting to robot server at {host}:{port}")

    _mock_cam: np.ndarray = None  # lazily allocated in mock mode

    # ─── HTTP helpers ──────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def _get_joints(self) -> np.ndarray:
        if self.mock:
            return self._current_joints.copy()
        try:
            r = _requests.get(self._url("/state"), timeout=2.0)
            r.raise_for_status()
            return np.array(r.json()["joints"], dtype=np.float32)
        except Exception as exc:
            print(f"[SO101RobotEnv] /state failed: {exc} — returning last known state")
            return self._current_joints.copy()

    def _send_joints(self, joints: np.ndarray) -> None:
        if self.mock:
            self._current_joints = joints.copy()
            return
        try:
            _requests.post(
                self._url("/step"),
                json={"joints": joints.tolist()},
                timeout=2.0,
            )
        except Exception as exc:
            print(f"[SO101RobotEnv] /step failed: {exc}")

    def _get_camera(self) -> np.ndarray:
        """Return a (CAM_H, CAM_W, CAM_C) float32 image in [0, 1].

        The robot HTTP server must expose:
            GET /camera → {"image": [[[r,g,b], ...], ...]}  (CAM_H × CAM_W × 3 uint8)
        """
        if self.mock:
            if self._mock_cam is None:
                self._mock_cam = np.zeros((CAM_H, CAM_W, CAM_C), dtype=np.float32)
            return self._mock_cam
        try:
            r = _requests.get(self._url("/camera"), timeout=2.0)
            r.raise_for_status()
            img = np.array(r.json()["image"], dtype=np.float32) / 255.0
            return img.reshape(CAM_H, CAM_W, CAM_C)
        except Exception as exc:
            print(f"[SO101RobotEnv] /camera failed: {exc} — returning blank frame")
            return np.zeros((CAM_H, CAM_W, CAM_C), dtype=np.float32)

    def _send_reset(self) -> None:
        if self.mock:
            self._current_joints = HOME_POSITION_DEG.copy()
            return
        try:
            # Send home position explicitly so the robot server never has to guess
            _requests.post(
                self._url("/reset"),
                json={"joints": HOME_POSITION_DEG.tolist()},
                timeout=5.0,
            )
        except Exception as exc:
            print(f"[SO101RobotEnv] /reset failed: {exc} — continuing anyway")

    # ─── action scaling ────────────────────────────────────────────────────────

    def _action_to_degrees(self, action_norm: np.ndarray) -> np.ndarray:
        """Map normalised [-1, 1] action → joint position targets in degrees.

        action = 0.0  →  home position  (untrained planner stays put)
        action = ±1.0 →  home ± half the physical joint range
        Clipped to JOINT_LOWER_DEG / JOINT_UPPER_DEG.
        """
        half = (JOINT_UPPER_DEG - JOINT_LOWER_DEG) / 2.0
        target = HOME_POSITION_DEG + action_norm * half
        return np.clip(target, JOINT_LOWER_DEG, JOINT_UPPER_DEG).astype(np.float32)

    def _clamp_delta(self, target_deg: np.ndarray) -> np.ndarray:
        """Limit per-step movement and block any joint already at its limit."""
        delta = np.clip(target_deg - self._current_joints, -MAX_DELTA_DEG, MAX_DELTA_DEG)
        # If a joint is at (or past) its upper limit, forbid further positive movement
        delta[self._current_joints >= JOINT_UPPER_DEG] = np.minimum(
            delta[self._current_joints >= JOINT_UPPER_DEG], 0.0
        )
        # If a joint is at (or past) its lower limit, forbid further negative movement
        delta[self._current_joints <= JOINT_LOWER_DEG] = np.maximum(
            delta[self._current_joints <= JOINT_LOWER_DEG], 0.0
        )
        return (self._current_joints + delta).astype(np.float32)

    # ─── Gymnasium API ─────────────────────────────────────────────────────────

    def _make_obs(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        cam = self._get_camera()
        return {
            OBS_KEY: joints.reshape(1, STATE_DIM).astype(np.float32),
            CAM_KEY: cam.reshape(1, CAM_H, CAM_W, CAM_C),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        t0 = time.time()
        self._send_reset()
        self._current_joints = self._get_joints()
        info: Dict[str, Any] = {
            "reset_time": round(time.time() - t0, 3),
            "joint_names": JOINT_NAMES,
        }
        return self._make_obs(self._current_joints), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        t0 = time.time()

        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
        target_deg = self._action_to_degrees(action_vec)
        target_deg = self._clamp_delta(target_deg)   # speed limit
        self._send_joints(target_deg)

        elapsed = time.time() - t0
        remaining = self._step_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        self._current_joints = self._get_joints()
        obs = self._make_obs(self._current_joints)

        info: Dict[str, Any] = {
            "joint_positions_deg": self._current_joints.tolist(),
            "action_norm": action_vec.tolist(),
            "mpail_env/step_time": round(time.time() - t0, 3),
        }
        # Reward is always 0 — MPAIL learns reward purely from demonstrations
        return obs, 0.0, False, False, info

    def read_robot_state(self) -> np.ndarray:
        """Return current joint positions in degrees (shape: (STATE_DIM,))."""
        return self._get_joints()

    def close(self) -> None:
        super().close()
