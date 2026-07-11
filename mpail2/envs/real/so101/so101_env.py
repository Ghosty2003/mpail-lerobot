"""SO-101 (SO-ARM101) base Gymnasium environment.

Communicates with the SO-101 robot-control server (so101_robot_server.py, run in
the `lerobot` conda env) over gRPC. The server directly wraps lerobot's SOFollower
driver and exposes two RPCs (see transport/so101_robot.proto):

    Reset (ResetRequest) -> RobotState
        Move the arm to HOME_POSITION_DEG and open the gripper.

    Step (StepRequest{joints_deg}) -> RobotState
        Execute a joint-position command and return the resulting state.

RobotState carries joints_deg (6 floats) plus both camera frames (raw uint8
bytes) in one message, so each step()/reset() call is a single round-trip
instead of four separate HTTP requests (state + step + camera + camera2).

Start the server (separate `lerobot` conda env, real hardware attached there):
    conda activate lerobot
    python so101_robot_server.py --robot_port /dev/ttyACM0 --robot_id Kid \\
        --cam_index /dev/video0 --cam2_serial <realsense_serial> --grpc_port 7070
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("so101_env")

from .robot_limits import (
    ACTION_DIM,
    CAM_C,
    CAM_H,
    CAM_KEY,
    CAM_W,
    CAM2_C,
    CAM2_H,
    CAM2_KEY,
    CAM2_W,
    CONTROL_FREQUENCY_HZ,
    DEFAULT_ROBOT_HOST,
    DEFAULT_ROBOT_PORT,
    EE_LOWER_M,
    EE_UPPER_M,
    GRIPPER_HALF_RANGE,
    HOME_POSITION_DEG,
    JOINT_LOWER_DEG,
    JOINT_NAMES,
    JOINT_UPPER_DEG,
    MAX_DELTA_M,
    MAX_EPISODE_STEPS,
    WRIST_ROLL_HALF_RANGE,
    EE_PROPRIO_DIM,
)
from ik_utils import fk as _fk, ik as _ik, joints_to_ee_proprio as _joints_to_ee_proprio

try:
    import grpc
    from transport import so101_robot_pb2, so101_robot_pb2_grpc
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False
    print("[WARNING] grpc / transport.so101_robot_pb2 not available — SO101RobotEnv will run in mock mode.")


# obs key used by planner_server.py (--obs_key default)
OBS_KEY = "observation.state"


class SO101RobotEnv(gym.Env):
    """Gymnasium env that wraps the SO-101 arm via the so101_robot_server.py gRPC service.

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
        speed_scale: float = 1.0,
        lpf_alpha: float = 0.6,
        reset_pause_seconds: float = 3.0,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.control_frequency = control_frequency
        self._step_dt = 1.0 / control_frequency
        self.max_episode_steps = max_episode_steps
        self.reset_pause_seconds = reset_pause_seconds
        # Alias: SO101RealWrapper reads env.unwrapped.max_episode_length (not
        # max_episode_steps) to size MPAIL2Runner's rollout length. Without this,
        # the wrapper's hasattr() check misses this attribute entirely and silently
        # falls back to the MAX_EPISODE_STEPS constant, ignoring whatever value was
        # actually requested (e.g. via --max_episode_steps).
        self.max_episode_length = max_episode_steps
        self.mock = mock or not _GRPC_AVAILABLE
        self.speed_scale = speed_scale
        # EMA weight on the new action (lower = smoother, more lag). Same role as
        # grpc_policy_server.py's --lpf_alpha: an untrained/highly-exploratory MPPI
        # policy's raw output is tanh-squashed noise that saturates near +-1 on most
        # dimensions most of the time, so without smoothing every step jerks near the
        # full (speed_scale-scaled) envelope — reducing speed_scale alone only shrinks
        # that envelope, it doesn't stop the action from constantly sitting at its edge.
        self._lpf_alpha = lpf_alpha
        self._prev_action_norm = None
        self._step_count = 0
        self._consecutive_rpc_failures = 0

        # runner.py reads these from env.unwrapped
        self.num_envs = 1

        self.observation_space = spaces.Dict({
            OBS_KEY: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(1, EE_PROPRIO_DIM),
                dtype=np.float32,
            ),
            CAM_KEY: spaces.Box(
                low=0.0, high=1.0,
                shape=(1, CAM_H, CAM_W, CAM_C),
                dtype=np.float32,
            ),
            CAM2_KEY: spaces.Box(
                low=0.0, high=1.0,
                shape=(1, CAM2_H, CAM2_W, CAM2_C),
                dtype=np.float32,
            ),
        })
        # Actions are normalised joint targets in [-1, 1] (planner outputs tanh)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, ACTION_DIM), dtype=np.float32
        )

        self._current_joints = HOME_POSITION_DEG.copy()
        self._channel = None
        self._stub = None

        if self.mock:
            print("[SO101RobotEnv] Running in mock mode — no gRPC calls will be made.")
        else:
            print(f"[SO101RobotEnv] Connecting to robot server at {host}:{port} (gRPC)")
            self._channel = grpc.insecure_channel(f"{host}:{port}")
            self._stub = so101_robot_pb2_grpc.SO101RobotStub(self._channel)

    _mock_cam:  np.ndarray = None  # lazily allocated in mock mode
    _mock_cam2: np.ndarray = None

    # ─── gRPC helpers ──────────────────────────────────────────────────────────

    def _decode_state(self, state: "so101_robot_pb2.RobotState") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RobotState -> (joints_deg (6,), cam (CAM_H,CAM_W,CAM_C) [0,1], cam2 (CAM2_H,CAM2_W,CAM2_C) [0,1])."""
        joints = np.array(state.joints_deg, dtype=np.float32)
        cam = (
            np.frombuffer(state.cam, dtype=np.uint8).reshape(CAM_H, CAM_W, CAM_C).astype(np.float32) / 255.0
            if state.cam else np.zeros((CAM_H, CAM_W, CAM_C), dtype=np.float32)
        )
        cam2 = (
            np.frombuffer(state.cam2, dtype=np.uint8).reshape(CAM2_H, CAM2_W, CAM2_C).astype(np.float32) / 255.0
            if state.cam2 else np.zeros((CAM2_H, CAM2_W, CAM2_C), dtype=np.float32)
        )
        return joints, cam, cam2

    def _mock_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mock_cam is None:
            self._mock_cam = np.zeros((CAM_H, CAM_W, CAM_C), dtype=np.float32)
        if self._mock_cam2 is None:
            self._mock_cam2 = np.zeros((CAM2_H, CAM2_W, CAM2_C), dtype=np.float32)
        return self._current_joints.copy(), self._mock_cam, self._mock_cam2

    def _rpc_reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        if self.mock:
            self._current_joints = HOME_POSITION_DEG.copy()
            return self._mock_state() + (True,)
        try:
            state = self._stub.Reset(so101_robot_pb2.ResetRequest(), timeout=8.0)
            return self._decode_state(state) + (True,)
        except grpc.RpcError as exc:
            logger.warning(f"Reset RPC failed ({exc.code()}: {exc.details()}) — continuing with last known state")
            return self._mock_state() + (False,)

    def _rpc_step(self, target_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        if self.mock:
            self._current_joints = target_deg.copy()
            return self._mock_state() + (True,)
        try:
            # 5s, not 2s: a real Step call round-trips send_action() + get_observation()
            # (both cameras) server-side; a slow camera/servo read shouldn't be mistaken
            # for a dead robot and silently fall back to stale state every subsequent step.
            state = self._stub.Step(
                so101_robot_pb2.StepRequest(joints_deg=target_deg.tolist()), timeout=5.0
            )
            return self._decode_state(state) + (True,)
        except grpc.RpcError as exc:
            logger.warning(f"Step RPC failed ({exc.code()}: {exc.details()}) — returning last known state")
            return self._mock_state() + (False,)

    # ─── action scaling ────────────────────────────────────────────────────────
    # Same Cartesian-delta + IK conversion as grpc_policy_server.py's
    # _project_action_to_bounds / _action_norm_to_joints, so this driver and the
    # gRPC policy server move the robot identically given the same checkpoint.

    def _project_action_to_bounds(
        self,
        action_norm: np.ndarray,
        current_ee: np.ndarray,
        current_arm_deg: np.ndarray,
        current_gripper_deg: float,
    ) -> np.ndarray:
        """Zero out action components that push against a hard boundary."""
        projected = action_norm.copy()

        for dim in range(3):
            if current_ee[dim] <= float(EE_LOWER_M[dim]) and projected[dim] < 0.0:
                projected[dim] = 0.0
            elif current_ee[dim] >= float(EE_UPPER_M[dim]) and projected[dim] > 0.0:
                projected[dim] = 0.0

        wr = current_arm_deg[4]
        if wr <= float(JOINT_LOWER_DEG[4]) and projected[3] < 0.0:
            projected[3] = 0.0
        elif wr >= float(JOINT_UPPER_DEG[4]) and projected[3] > 0.0:
            projected[3] = 0.0

        g = current_gripper_deg
        if g <= float(JOINT_LOWER_DEG[5]) and projected[4] < 0.0:
            projected[4] = 0.0
        elif g >= float(JOINT_UPPER_DEG[5]) and projected[4] > 0.0:
            projected[4] = 0.0

        return projected

    def _action_norm_to_joints(self, action_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a 5-dim normalised action [x,y,z,wrist_roll,gripper] to 6 joint degrees via IK.

        Returns (joints_deg (6,), effective_action_norm (5,)) — effective_action_norm is the
        post-boundary-projection action, used by step() as the LPF reference for next step.
        """
        current_arm_deg = self._current_joints[:5].copy()
        current_ee = _fk(current_arm_deg)
        current_gripper_deg = float(self._current_joints[5])

        action_norm = self._project_action_to_bounds(
            action_norm, current_ee, current_arm_deg, current_gripper_deg
        )

        max_d = float(MAX_DELTA_M * self.speed_scale)
        delta_xyz = action_norm[:3].astype(np.float32) * max_d
        target_xyz = np.clip(current_ee + delta_xyz, EE_LOWER_M, EE_UPPER_M)

        arm_deg = _ik(target_xyz, initial_arm_deg=current_arm_deg)
        arm_deg = np.clip(arm_deg, JOINT_LOWER_DEG[:5], JOINT_UPPER_DEG[:5])

        wrist_delta_deg = float(action_norm[3]) * float(WRIST_ROLL_HALF_RANGE) * self.speed_scale
        arm_deg[4] = float(np.clip(current_arm_deg[4] + wrist_delta_deg,
                                   JOINT_LOWER_DEG[4], JOINT_UPPER_DEG[4]))

        gripper_delta_deg = float(action_norm[4]) * float(GRIPPER_HALF_RANGE) * self.speed_scale
        gripper_deg = float(np.clip(current_gripper_deg + gripper_delta_deg,
                                    JOINT_LOWER_DEG[5], JOINT_UPPER_DEG[5]))

        return np.append(arm_deg, gripper_deg).astype(np.float32), action_norm

    # ─── Gymnasium API ─────────────────────────────────────────────────────────

    def _make_obs(self, joints: np.ndarray, cam: np.ndarray, cam2: np.ndarray) -> Dict[str, np.ndarray]:
        ee_proprio = _joints_to_ee_proprio(joints)
        return {
            OBS_KEY:  ee_proprio.reshape(1, EE_PROPRIO_DIM).astype(np.float32),
            CAM_KEY:  cam.reshape(1, CAM_H, CAM_W, CAM_C),
            CAM2_KEY: cam2.reshape(1, CAM2_H, CAM2_W, CAM2_C),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        t0 = time.time()
        self._prev_action_norm = None  # don't carry LPF state across episode boundaries
        self._step_count = 0
        self._consecutive_rpc_failures = 0
        self._current_joints, cam, cam2, ok = self._rpc_reset()
        logger.info(
            f"[reset] ok={ok}  joints_deg={np.round(self._current_joints, 2).tolist()}  "
            f"took={time.time() - t0:.2f}s"
        )
        if self.reset_pause_seconds > 0:
            logger.info(f"[reset] pausing {self.reset_pause_seconds:.1f}s for the arm to settle at home...")
            time.sleep(self.reset_pause_seconds)
        info: Dict[str, Any] = {
            "reset_time": round(time.time() - t0, 3),
            "joint_names": JOINT_NAMES,
        }
        return self._make_obs(self._current_joints, cam, cam2), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        t0 = time.time()

        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_vec = raw_action.copy()

        # LPF on xyz + wrist_roll only (indices 0-3); gripper (index 4) is excluded so it
        # always tracks the raw commanded value, same convention as grpc_policy_server.py.
        if self._prev_action_norm is not None:
            gripper_val = action_vec[4]
            action_vec = (self._lpf_alpha * action_vec
                          + (1 - self._lpf_alpha) * self._prev_action_norm)
            action_vec[4] = gripper_val
            action_vec = np.clip(action_vec, -1.0, 1.0)

        prev_joints = self._current_joints.copy()
        target_deg, effective_norm = self._action_norm_to_joints(action_vec)
        self._prev_action_norm = effective_norm.copy()
        self._current_joints, cam, cam2, ok = self._rpc_step(target_deg)
        obs = self._make_obs(self._current_joints, cam, cam2)

        if ok:
            self._consecutive_rpc_failures = 0
        else:
            self._consecutive_rpc_failures += 1
            if self._consecutive_rpc_failures >= 3:
                logger.error(
                    f"[step {self._step_count}] Step RPC has failed "
                    f"{self._consecutive_rpc_failures} times in a row — joints have not "
                    f"actually updated since. Check so101_robot_server.py (camera/servo hang?) "
                    f"or raise the RPC timeout further if this is just slow, not stuck."
                )

        ee_xyz = _fk(self._current_joints[:5])
        moved = self._current_joints - prev_joints
        logger.info(
            f"[step {self._step_count}] ok={ok}  raw={np.round(raw_action, 3).tolist()}  "
            f"eff={np.round(effective_norm, 3).tolist()}  "
            f"target_deg={np.round(target_deg, 2).tolist()}  "
            f"actual_deg={np.round(self._current_joints, 2).tolist()}  "
            f"delta_deg={np.round(moved, 2).tolist()}  "
            f"EE=({ee_xyz[0]:+.4f},{ee_xyz[1]:+.4f},{ee_xyz[2]:+.4f})  "
            f"elapsed={time.time() - t0:.3f}s"
        )
        self._step_count += 1

        elapsed = time.time() - t0
        remaining = self._step_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        info: Dict[str, Any] = {
            "joint_positions_deg": self._current_joints.tolist(),
            "action_norm": action_vec.tolist(),
            "rpc_ok": ok,
            "mpail_env/step_time": round(time.time() - t0, 3),
        }
        # Reward is always 0 — MPAIL learns reward purely from demonstrations
        return obs, 0.0, False, False, info

    def read_robot_state(self) -> np.ndarray:
        """Return current joint positions in degrees (shape: (STATE_DIM,))."""
        return self._current_joints.copy()

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
        super().close()
