"""
grpc_policy_server.py — MPAIL2 gRPC inference + online learning server for SO-101.

Two modes
─────────
Demo collection (no demo file required):
    python grpc_policy_server.py --mock --collect_dir ./raw_demos2 --flush_every 50

    Returns home-position actions so the robot holds still while you move it manually
    or via a leader arm.  Every (obs_t, obs_t+1) pair is saved to .npz files.

    After collecting, convert to .pt:
        python convert.py --dirs raw_demos2 --out raw_demos2_master.pt

Online RL (requires a demo .pt file):
    python grpc_policy_server.py --demo_path raw_demos2_master.pt

Robot client (lerobot env, either mode):
    python -m lerobot.async_inference.robot_client \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyUSB0 \\
        --robot.id=my_robot \\
        --robot.cameras="{cam: {type: opencv, index_or_path: 0, width: 84, height: 84, fps: 30}}" \\
        --server_address=127.0.0.1:8080 \\
        --task="pick up the cup" \\
        --actions_per_chunk=1 \\
        --max_episode_steps=300 \
        --reset_pause_seconds=10
"""

import argparse
import io
import logging
import os
import pickle  # nosec
import time

import torch.nn.functional as F
import sys
import threading
import types
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional

import grpc
import numpy as np
import torch

torch.set_float32_matmul_precision('high')  # enable TF32 tensor cores on Ampere+ GPUs

import warnings
warnings.filterwarnings("ignore", message="Not enough SMs to use max_autotune_gemm")
warnings.filterwarnings("ignore", message=".*pow_by_natural.*")

from transport import services_pb2, services_pb2_grpc

from mpail2.envs.real.so101 import (
    SO101RealEnvArgs, make_so101_env, OBS_KEY, STATE_DIM, ACTION_DIM, EE_PROPRIO_DIM,
    HOME_POSITION_DEG,
)
from mpail2.envs.real.so101.robot_limits import (
    CAM_KEY, CAM_H, CAM_W, CAM_C,
    CAM2_KEY, CAM2_H, CAM2_W, CAM2_C,
    EE_LOWER_M, EE_UPPER_M, MAX_DELTA_M,
    HOME_WRIST_ROLL_DEG, WRIST_ROLL_HALF_RANGE,
    HOME_GRIPPER_DEG, GRIPPER_HALF_RANGE,
    JOINT_LOWER_DEG, JOINT_UPPER_DEG,
)
from ik_utils import fk as _fk, fk_pose as _fk_pose, ik as _ik

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("grpc_policy_server")


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA DISPLAY — live OpenCV window updated from a background thread
# ─────────────────────────────────────────────────────────────────────────────

_CAM_SAVE_PATH = "cameras_latest.png"

def _update_camera_display(image_arrays: dict) -> None:
    """Save both camera images side-by-side to cameras_latest.png.

    Open this file in VSCode — it auto-refreshes as the file updates.
    Layout: RealSense (cam2) on the left, wrist cam (cam) on the right.
    """
    try:
        import cv2
    except ImportError:
        return
    frames = []
    for key in ("cam2", "cam"):
        img = image_arrays.get(key)
        if img is None:
            continue
        img = np.array(img, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        if img_u8.shape[2] == 3:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        img_u8 = cv2.resize(img_u8, (320, 240))
        frames.append(img_u8)
    if not frames:
        return
    composed = np.concatenate(frames, axis=1)
    cv2.imwrite(_CAM_SAVE_PATH, composed)


# ─────────────────────────────────────────────────────────────────────────────
# PICKLE COMPAT — deserialize lerobot TimedObservation, send back TimedAction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TimedData:
    timestamp: float
    timestep: int
    def get_timestamp(self): return self.timestamp
    def get_timestep(self):  return self.timestep

@dataclass
class TimedObservation(_TimedData):
    observation: dict
    must_go: bool = False
    def get_observation(self): return self.observation

@dataclass
class TimedAction(_TimedData):
    action: torch.Tensor
    def get_action(self): return self.action

TimedAction.__module__   = "lerobot.async_inference.helpers"
TimedAction.__qualname__ = "TimedAction"
_TimedData.__module__    = "lerobot.async_inference.helpers"
_TimedData.__qualname__  = "TimedData"

class _LeRobotUnpickler(pickle.Unpickler):
    _MAP = {
        ("lerobot.async_inference.helpers", "TimedObservation"): TimedObservation,
        ("lerobot.async_inference.helpers", "TimedData"):        _TimedData,
    }
    def find_class(self, module, name):
        return self._MAP.get((module, name)) or super().find_class(module, name)

def _loads(data: bytes):
    return _LeRobotUnpickler(io.BytesIO(data)).load()

_lhm = types.ModuleType("lerobot.async_inference.helpers")
_lhm.TimedData = _TimedData; _lhm.TimedAction = TimedAction; _lhm.TimedObservation = TimedObservation
sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
sys.modules.setdefault("lerobot.async_inference", types.ModuleType("lerobot.async_inference"))
sys.modules["lerobot.async_inference.helpers"] = _lhm


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION / ACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_raw_obs(raw_obs: dict) -> tuple[np.ndarray, dict]:
    joint_vals = []
    image_arrays = {}
    for key, value in raw_obs.items():
        if key in ("task", "teleop_action"):
            continue
        arr = np.array(value, dtype=np.float32)
        if arr.ndim == 3:
            # Camera image (H, W, C); strip lerobot prefix if present
            cam_key = key.removeprefix("observation.images.")
            image_arrays[cam_key] = arr
        else:
            # Scalar motor key ("shoulder_pan.pos", ...) or full "observation.state" array
            joint_vals.extend(arr.flatten().tolist())
    return np.array(joint_vals, dtype=np.float32), image_arrays


def _joints_to_ee_proprio(joint_state: np.ndarray) -> np.ndarray:
    """Convert 6-dim joint state (degrees) to 13-dim proprioception.

    Returns [ee_x, ee_y, ee_z, qx, qy, qz, qw, j0..j5] — matches Franka convention.
    """
    xyz, quat = _fk_pose(joint_state[:5])              # xyz (3,), quat [x,y,z,w] (4,)
    return np.concatenate([xyz, quat, joint_state], dtype=np.float32)  # (13,)


def _build_obs_dict(joint_state: np.ndarray, image_arrays: dict, device: str) -> dict:
    ee_proprio = _joints_to_ee_proprio(joint_state)
    result = {OBS_KEY: torch.from_numpy(ee_proprio).unsqueeze(0).to(device)}
    for cam_name, img_arr in image_arrays.items():
        img = img_arr.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        if img.shape[0] != CAM_H or img.shape[1] != CAM_W:
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            t = F.interpolate(t, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False)
            img = t.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
        result[cam_name] = torch.from_numpy(img).unsqueeze(0).to(device)
    return result


def _project_action_to_bounds(
    action_norm: np.ndarray,
    current_ee: np.ndarray,
    current_arm_deg: np.ndarray,
    current_gripper_deg: float,
) -> np.ndarray:
    """Zero out action components that push against a hard boundary (same as Franka).

    EE xyz (indices 0-2): zeroed when at workspace bounds.
    Wrist roll (index 3): zeroed when at joint limits.
    Gripper (index 4): zeroed at limits.
    """
    projected = action_norm.copy()

    # EE xyz axes: zero out push against boundary (same as Franka)
    for dim in range(3):
        if current_ee[dim] <= float(EE_LOWER_M[dim]) and projected[dim] < 0.0:
            projected[dim] = 0.0
        elif current_ee[dim] >= float(EE_UPPER_M[dim]) and projected[dim] > 0.0:
            projected[dim] = 0.0

    # Wrist roll: zero at joint limits
    wr = current_arm_deg[4]
    if wr <= float(JOINT_LOWER_DEG[4]) and projected[3] < 0.0:
        projected[3] = 0.0
    elif wr >= float(JOINT_UPPER_DEG[4]) and projected[3] > 0.0:
        projected[3] = 0.0

    # Gripper: zero at limits (can't meaningfully reflect open/close)
    g = current_gripper_deg if current_gripper_deg is not None else float(HOME_GRIPPER_DEG)
    if g <= float(JOINT_LOWER_DEG[5]) and projected[4] < 0.0:
        projected[4] = 0.0
    elif g >= float(JOINT_UPPER_DEG[5]) and projected[4] > 0.0:
        projected[4] = 0.0

    return projected


def _action_norm_to_joints(
    action_norm: np.ndarray,        # shape (5,): [x_n, y_n, z_n, wrist_roll_n, gripper_n] in [-1,1]
    current_ee: np.ndarray,         # current EE xyz (m), delta base position
    current_arm_deg: np.ndarray,    # current 5 arm joints (deg), IK warm-start
    speed_scale: float = 1.0,
    current_gripper_deg: float = None,  # current gripper angle (deg), delta base
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a 5-dim normalised action to 6 joint-degree targets via IK.

    Applies boundary projection first: components blocked by workspace / joint limits
    are zeroed so the robot never pushes against a wall.

    Returns (joints_deg shape (6,), effective_action_norm shape (5,)).
    effective_action_norm is the post-projection action that should be stored in the
    replay buffer so the dynamics model learns actual boundary behaviour.
    """
    # ── Project to feasible region first ──
    action_norm = _project_action_to_bounds(
        action_norm, current_ee, current_arm_deg, current_gripper_deg
    )

    # ── Arm xyz — delta ──
    max_d = float(MAX_DELTA_M * speed_scale)
    delta_xyz = action_norm[:3].astype(np.float32) * max_d
    target_xyz = np.clip(current_ee + delta_xyz, EE_LOWER_M, EE_UPPER_M)

    arm_deg = _ik(target_xyz, initial_arm_deg=current_arm_deg)
    arm_deg = np.clip(arm_deg, JOINT_LOWER_DEG[:5], JOINT_UPPER_DEG[:5])

    # ── Wrist roll — delta ──
    wrist_delta_deg = float(action_norm[3]) * float(WRIST_ROLL_HALF_RANGE) * speed_scale
    arm_deg[4] = float(np.clip(current_arm_deg[4] + wrist_delta_deg,
                               JOINT_LOWER_DEG[4], JOINT_UPPER_DEG[4]))

    # ── Gripper — delta ──
    base_g = float(current_gripper_deg) if current_gripper_deg is not None else HOME_GRIPPER_DEG
    gripper_delta_deg = float(action_norm[4]) * float(GRIPPER_HALF_RANGE) * speed_scale
    gripper_deg = float(np.clip(base_g + gripper_delta_deg, JOINT_LOWER_DEG[5], JOINT_UPPER_DEG[5]))

    return np.append(arm_deg, gripper_deg).astype(np.float32), action_norm


# ─────────────────────────────────────────────────────────────────────────────
# DEMO RECORDER  (same logic as planner_server.py _record_step / _flush)
# ─────────────────────────────────────────────────────────────────────────────

class DemoRecorder:
    """Buffers (obs_t, obs_t+1) pairs and flushes to .npz files."""

    def __init__(self, collect_dir: Path, flush_every: int = 200):
        self.collect_dir = collect_dir
        self.flush_every = flush_every
        collect_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._obs_buf:         List[np.ndarray] = []
        self._next_obs_buf:    List[np.ndarray] = []
        self._image_bufs:      Dict[str, List[np.ndarray]] = {}
        self._image_next_bufs: Dict[str, List[np.ndarray]] = {}
        self._pending_obs:     Optional[np.ndarray] = None
        self._pending_images:  Dict[str, np.ndarray] = {}
        self._file_idx = 0

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.max() <= 1.0:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img.clip(0, 255).astype(np.uint8)

    def record(self, obs: np.ndarray, images: Dict[str, np.ndarray]):
        images_u8 = {k: self._to_uint8(v) for k, v in images.items()}
        with self._lock:
            if self._pending_obs is not None:
                self._obs_buf.append(self._pending_obs)
                self._next_obs_buf.append(obs.copy())
                for cam, img in self._pending_images.items():
                    self._image_bufs.setdefault(cam, []).append(img)
                    self._image_next_bufs.setdefault(cam, []).append(
                        images_u8.get(cam, img).copy()
                    )

            self._pending_obs    = obs.copy()
            self._pending_images = images_u8

            if len(self._obs_buf) >= self.flush_every:
                self._flush()

    def flush(self):
        with self._lock:
            self._flush()

    def reset_pending(self):
        """Discard any half-open transition so the next call to record() starts fresh."""
        with self._lock:
            self._pending_obs    = None
            self._pending_images = {}

    def _flush(self):
        if not self._obs_buf:
            return
        obs_arr  = np.stack(self._obs_buf)
        nobs_arr = np.stack(self._next_obs_buf)
        save_data = {OBS_KEY: np.stack([obs_arr, nobs_arr], axis=1)}  # (N, 2, state_dim)
        for cam in self._image_bufs:
            imgs  = np.stack(self._image_bufs[cam])
            nimgs = np.stack(self._image_next_bufs[cam])
            save_data[cam] = np.stack([imgs, nimgs], axis=1)           # (N, 2, H, W, C)

        path = self.collect_dir / f"traj_{self._file_idx:04d}.npz"
        np.savez(str(path), **save_data)
        n = len(self._obs_buf)
        cams = list(self._image_bufs.keys())
        logger.info(f"[recorder] Saved {n} steps → {path}  cameras={cams or 'none'}")

        self._obs_buf.clear(); self._next_obs_buf.clear()
        self._image_bufs.clear(); self._image_next_bufs.clear()
        self._file_idx += 1
        self._pending_obs    = None
        self._pending_images = {}


# ─────────────────────────────────────────────────────────────────────────────
# gRPC SERVICER
# ─────────────────────────────────────────────────────────────────────────────

class MPAILServicer(services_pb2_grpc.AsyncInferenceServicer):

    def __init__(
        self,
        runner,
        device: str,
        mock: bool,
        recorder: Optional[DemoRecorder],
        speed_scale: float = 1.0,
        eval_mode: bool = False,
        gripper_update_every: int = 5,
        lpf_alpha: float = 0.6,
        start_episode: int = 0,
        max_episode_steps: int = 200,
        show_cameras: bool = False,
    ):
        self.runner      = runner      # None in mock mode
        self.device      = device
        self.mock        = mock
        self.eval_mode   = eval_mode
        self.recorder    = recorder
        self.speed_scale = speed_scale
        self.gripper_update_every = max(1, gripper_update_every)
        self.max_episode_steps = max_episode_steps
        self.show_cameras = show_cameras
        if show_cameras:
            logger.info(f"[cameras] Saving live frames to {_CAM_SAVE_PATH} — open in VSCode to watch")

        self._lock = threading.Lock()
        self._obs_queue: Queue = Queue(maxsize=1)
        self.shutdown_event = threading.Event()

        self._prev_obs_dict  = None
        self._transition_open = False  # set after learner.act(), cleared after process_env_step
        self._episode_steps  = 0
        self._episode_count  = start_episode
        self._prev_timestep  = -1
        self._recording_paused = False   # True while waiting for Enter between episodes
        self._actions_per_chunk = 1      # updated from SendPolicyInstructions

        # Exploration is handled by MPPI sampling (policy_proportion=0.05, max_std=2.0)
        # — same as Franka. No episode-level noise decay needed.
        self._prev_action_norm  = None
        self._lpf_alpha         = lpf_alpha  # EMA weight on new action (lower = smoother, more lag)
        self._last_gripper_deg  = None
        self._prev_gripper_norm = None       # separate LPF state for gripper (only steps when gripper updates)
        self._gripper_chunk_count = 0
        self._disc_rewards_this_ep: List[float] = []

    def Ready(self, request, context):
        # Kick off end-of-episode work (save / train) in a background thread so the
        # client gets this response immediately and the arm can keep moving.
        # _recording_paused is set to True inside _on_episode_end() before any slow
        # work starts, so incoming observations are discarded until the next episode
        # is ready to record.
        if self._episode_steps > 0 and not self.eval_mode:
            logger.info(f"Ready: starting episode-end work in background (ep {self._episode_count + 1}, {self._episode_steps} steps)...")
            threading.Thread(target=self._on_episode_end, daemon=True).start()
        logger.info(f"Robot client connected: {context.peer()}")
        self.shutdown_event.clear()
        self._obs_queue = Queue(maxsize=1)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        try:
            policy_config = pickle.loads(request.data)  # nosec
            self._actions_per_chunk = int(getattr(policy_config, "actions_per_chunk", 1))
            logger.info(f"Policy instructions: actions_per_chunk={self._actions_per_chunk}")
        except Exception as e:
            logger.warning(f"Could not parse policy instructions: {e}")
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):
        try:
            buf = io.BytesIO()
            n_chunks = 0
            for chunk in request_iterator:
                state = chunk.transfer_state
                if state == services_pb2.TransferState.TRANSFER_BEGIN:
                    buf.seek(0); buf.truncate(0)
                buf.write(chunk.data)
                n_chunks += 1
                if state == services_pb2.TransferState.TRANSFER_END:
                    break
            data = buf.getvalue()
            logger.debug(f"SendObservations: received {n_chunks} chunk(s), {len(data)} bytes")
        except Exception as e:
            logger.exception(f"SendObservations chunk read error: {e}")
            return services_pb2.Empty()

        # While paused between episodes, discard immediately — arm movement is
        # handled client-side (leader→follower), so we don't need to queue or record.
        if self._recording_paused:
            logger.debug("SendObservations: discarded (recording paused)")
            return services_pb2.Empty()

        try:
            timed_obs: TimedObservation = _loads(data)
        except Exception as e:
            logger.exception(f"SendObservations deserialize error: {e}  data[:64]={data[:64]!r}")
            return services_pb2.Empty()

        try:
            timestep = timed_obs.get_timestep()
            joint_state, image_arrays = _parse_raw_obs(timed_obs.get_observation())

            if self._episode_steps == 0 and self._prev_obs_dict is None:
                cam_info = {k: v.shape for k, v in image_arrays.items()}
                logger.info(f"[obs check] joints={joint_state.shape} cameras={cam_info}")

            if self.show_cameras:
                _update_camera_display(image_arrays)

            if self._episode_steps % 10 == 0:
                from mpail2.envs.real.so101.robot_limits import JOINT_NAMES
                joint_str = "  ".join(f"{n}={v:.2f}" for n, v in zip(JOINT_NAMES, joint_state))
                logger.info(f"[joints] {joint_str}")

            self._prev_timestep = timestep

            # Record demo transition
            if self.recorder is not None:
                self.recorder.record(joint_state, image_arrays)

            # Push to obs queue so GetActions can respond
            if self._obs_queue.full():
                try: self._obs_queue.get_nowait()
                except Empty: pass
            self._obs_queue.put((timestep, timed_obs.get_timestamp(), joint_state, image_arrays))

        except Exception as e:
            logger.exception(f"SendObservations processing error: {e}")

        return services_pb2.Empty()

    def GetActions(self, request, context):
        try:
            timestep, timestamp, joint_state, image_arrays = self._obs_queue.get(timeout=1.0)
        except Empty:
            return services_pb2.Actions(data=b"")

        if self.mock:
            # Echo current joints back — robot actively holds wherever it already is.
            action_deg = joint_state.copy()
            timed_actions = [TimedAction(
                timestamp=timestamp, timestep=timestep,
                action=torch.tensor(action_deg, dtype=torch.float32),
            )]
        else:
            t_act_start = time.time()
            obs_dict = _build_obs_dict(joint_state, image_arrays, self.device)
            with self._lock:
                self.runner.learner.eval()

                # Close the transition opened by the previous learner.act() call.
                # Doing this in GetActions keeps observation/action bookkeeping on
                # one RPC thread instead of racing SendObservations against act().
                if not self.eval_mode and self._prev_obs_dict is not None and self._transition_open:
                    reward = torch.tensor([0.0], device=self.device)
                    done = torch.tensor([0], dtype=torch.long, device=self.device)
                    buffer_full = self._episode_steps >= min(self.max_episode_steps, self.runner.learner.storage.num_steps_per_env)
                    if not buffer_full:
                        self.runner.learner.process_env_step(reward, done, {}, obs_dict)
                        self._episode_steps += 1
                    else:
                        logger.debug(f"Buffer full at step {self._episode_steps} — skipping transition")
                    self._transition_open = False

                    with torch.no_grad():
                        z  = self.runner.learner._encoder(self._prev_obs_dict)
                        zn = self.runner.learner._encoder(obs_dict)
                        disc_r = self.runner.learner._reward(z, zn).item()
                    self._disc_rewards_this_ep.append(disc_r)
                    logger.info(
                        f"[ep {self._episode_count + 1} step {self._episode_steps}]"
                        f"  disc_reward={disc_r:.4f}"
                    )

                with torch.no_grad():
                    self.runner.learner.act(obs_dict)   # runs MPPI, fills _opt_controls
                self._prev_obs_dict = obs_dict
                self._transition_open = not self.eval_mode

            # Full planned trajectory: (num_timesteps, 5) — [x_n, y_n, z_n, wrist_roll_n, gripper_n]
            traj_norm = self.runner.learner.planner._opt_controls[0].detach().cpu().numpy()

            # Build action chunk.
            # current_arm_deg: 5 arm joints used as IK warm-start (propagated per step).
            # current_ee:      EE xyz used for Cartesian speed limiting (propagated per step).
            timed_actions = []
            current_arm_deg = joint_state[:5].copy()
            current_ee      = _fk(current_arm_deg)
            # Log MPPI output every 20 steps to diagnose action direction
            if self._episode_steps % 20 == 0:
                _raw = traj_norm[0]  # first planned step, pre-LPF
                logger.info(
                    f"[mppi] x={_raw[0]:+.3f}  y={_raw[1]:+.3f}  z={_raw[2]:+.3f}  "
                    f"wrist={_raw[3]:+.3f}  grip={_raw[4]:+.3f}  "
                    f"EE_y={current_ee[1]:+.4f}m  EE_z={current_ee[2]:+.4f}m  "
                    f"pan={joint_state[0]:+.1f}deg"
                )
            update_gripper = (
                self._last_gripper_deg is None
                or self._gripper_chunk_count % self.gripper_update_every == 0
            )
            for t_offset, step_norm in enumerate(traj_norm):
                action_norm = step_norm.copy()   # shape (5,): [x, y, z, wrist_roll, gripper]
                if t_offset == 0:
                    # LPF on xyz + wrist_roll only (indices 0-3); gripper is excluded
                    # because it has its own hold mechanism (update_gripper / _last_gripper_deg).
                    if self._prev_action_norm is not None:
                        gripper_val = action_norm[4]   # save before blending
                        action_norm = (self._lpf_alpha * action_norm
                                       + (1 - self._lpf_alpha) * self._prev_action_norm)
                        action_norm[4] = gripper_val   # restore unblended gripper
                        action_norm = np.clip(action_norm, -1.0, 1.0)

                    if update_gripper:
                        self._prev_gripper_norm = float(action_norm[4])
                    else:
                        # Hold window: restore gripper to last committed norm
                        if self._prev_gripper_norm is not None:
                            action_norm[4] = self._prev_gripper_norm

                    # IK: 5-dim norm → 6 joint degrees (returns effective post-projection action)
                    step_deg, effective_norm = _action_norm_to_joints(
                        action_norm, current_ee, current_arm_deg, self.speed_scale,
                        current_gripper_deg=self._last_gripper_deg,
                    )

                    self._prev_action_norm = effective_norm.copy()

                    # Sync replay buffer: store effective (post-projection) action so the
                    # dynamics model learns that boundary = no movement in that direction.
                    if not self.eval_mode:
                        stored_norm = effective_norm.copy()
                        if not update_gripper:
                            # Gripper holding: store zero delta so dynamics sees no change.
                            stored_norm[4] = 0.0
                        self.runner.learner.transition.actions = torch.tensor(
                            stored_norm, dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                else:
                    step_deg, _ = _action_norm_to_joints(
                        action_norm, current_ee, current_arm_deg, self.speed_scale,
                        current_gripper_deg=self._last_gripper_deg,
                    )

                if update_gripper and t_offset == 0:
                    # Only lock in the gripper from the executed step (t_offset=0).
                    # Later horizon steps are not sent to the robot and must not
                    # overwrite _last_gripper_deg with their raw (unfiltered) values.
                    self._last_gripper_deg = float(step_deg[5])
                elif not update_gripper:
                    step_deg[5] = self._last_gripper_deg

                timed_actions.append(TimedAction(
                    timestamp=timestamp,
                    timestep=timestep + t_offset,
                    action=torch.tensor(step_deg, dtype=torch.float32),
                ))
                # Propagate for next step's speed limiting and IK warm-start
                current_arm_deg = step_deg[:5].copy()
                current_ee      = _fk(current_arm_deg)
                current = step_deg  # propagate so next step's delta is from the right base
            self._gripper_chunk_count += 1

            # Respect the client's requested chunk size so the observation rate
            # stays in sync with the control rate (actions_per_chunk=1 → 1 obs/step).
            timed_actions = timed_actions[:self._actions_per_chunk]

            logger.info(
                f"act() → {len(timed_actions)}-step chunk (of {len(traj_norm)} planned)  "
                f"elapsed={time.time()-t_act_start:.2f}s"
            )

        return services_pb2.Actions(data=pickle.dumps(timed_actions))  # nosec

    def _on_episode_end(self):
        if self.mock:
            self._episode_count += 1
            if self.recorder is not None:
                # Pause recording immediately so observations sent while saving are discarded.
                self._recording_paused = True
                ep = self._episode_count
                def _save_and_wait():
                    self.recorder.flush()  # slow disk write — runs in background
                    print(
                        f"\n── Episode {ep} saved ──────────────────────────\n"
                        f"   Press Enter to start recording episode {ep + 1} "
                        f"(Ctrl-C to stop)...",
                        flush=True,
                    )
                    self._wait_for_enter()
                threading.Thread(target=_save_and_wait, daemon=True).start()
            return

        self._recording_paused = True   # discard observations during training
        train_stats = {}
        with self._lock:
            self._episode_count += 1
            try:
                self.runner.learner.train()
                # update() calls storage.process_replay_buffer() first, which is what
                # moves the current rollout into the replay buffer before training begins.
                train_stats = self.runner.learner.update(iteration=self._episode_count)
                self.runner.learner.eval()
            except Exception as exc:
                logger.error(f"[ep {self._episode_count}] update() failed: {exc}")
                self.runner.learner.eval()
            _done_idx = torch.zeros(1, dtype=torch.long, device=self.device)
            self.runner.learner.planner.reset(reset_inds=_done_idx)
            self.runner.learner.storage.clear()

        logger.info(f"[ep {self._episode_count}] training done")

        disc_rewards = self._disc_rewards_this_ep[:]
        episode_steps = self._episode_steps

        self._episode_steps = 0
        self._prev_obs_dict = None
        self._transition_open = False
        self._prev_action_norm  = None
        self._last_gripper_deg  = None
        self._prev_gripper_norm = None
        self._gripper_chunk_count = 0
        self._disc_rewards_this_ep = []

        if train_stats:
            dyn    = train_stats.get('Dyn/mean_loss', 0)
            reward = train_stats.get('Reward/mean_gen_reward', 0)
            value  = train_stats.get('Value/mean_loss', 0)
            policy = train_stats.get('Policy/mean_loss', 0)
            logger.info(
                f"[ep {self._episode_count}] "
                f"dyn={dyn:.4f}  reward={reward:.4f}  "
                f"value={value:.4f}  policy={policy:.4f}"
            )
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    log_dict = {f"train/{k}": v for k, v in train_stats.items()
                                if isinstance(v, (int, float))}
                    log_dict["episode"] = self._episode_count
                    log_dict["episode/steps"] = episode_steps
                    if disc_rewards:
                        log_dict["episode/disc_reward_mean"] = float(np.mean(disc_rewards))
                        log_dict["episode/disc_reward_min"]  = float(np.min(disc_rewards))
                        log_dict["episode/disc_reward_max"]  = float(np.max(disc_rewards))

                    # MPPI statistics — action distribution mean & std per dimension
                    with torch.no_grad():
                        mppi_stats = self.runner.learner.planner.compute_stats()
                    log_dict.update({k: float(v) for k, v in mppi_stats.items()
                                     if isinstance(v, (int, float, torch.Tensor))
                                     and (not isinstance(v, torch.Tensor) or v.numel() == 1)})
                    # Per-dim action mean (first planned step)
                    opt = self.runner.learner.planner._opt_controls[0, 0].detach().cpu()  # (nu,)
                    dim_names = ["x", "y", "z", "wrist", "grip"]
                    for i, name in enumerate(dim_names[:opt.shape[0]]):
                        log_dict[f"MPPI/action_mean_{name}"] = float(opt[i])
                    # Per-dim action std
                    iter_std = self.runner.learner.planner.sampling._iter_std[0, 0].detach().cpu()  # (nu,)
                    for i, name in enumerate(dim_names[:iter_std.shape[0]]):
                        log_dict[f"MPPI/action_std_{name}"] = float(iter_std[i])

                    _wandb.log(log_dict, step=self._episode_count)
            except ImportError:
                pass
        if self._episode_count % 20 == 0:
            self.runner.current_learning_iteration = self._episode_count
            self.runner.save(postfix=f"ep{self._episode_count}")
        self._recording_paused = False   # ready to record next episode

    def _wait_for_enter(self):
        """Block in a background thread until the user presses Enter, then resume recording."""
        try:
            input()
        except EOFError:
            pass
        self.recorder.reset_pending()   # clean slate for the new episode
        self._recording_paused = False
        print(f"   Recording episode {self._episode_count + 1} ...", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def serve():
    parser = argparse.ArgumentParser(description="MPAIL2 gRPC policy server for SO-101")
    parser.add_argument("--demo_path",    default=None,
                        help="Path to .pt demo file. Required unless --mock is set.")
    parser.add_argument("--demo_subsample", type=int, default=30,
                        help="Keep every Nth demo transition to match online control frequency. "
                             "Demos collected at 30Hz with online at 1Hz → use 30 (default). "
                             "Set to 1 to disable sub-sampling.")
    parser.add_argument("--mock",         action="store_true",
                        help="Skip runner; return home-position actions. Use with --collect_dir to record demos.")
    parser.add_argument("--collect_dir",  default=None,
                        help="Directory to save recorded (obs, next_obs) .npz files.")
    parser.add_argument("--flush_every",  type=int, default=200,
                        help="Flush buffer to disk every N steps (default: 200).")
    parser.add_argument("--port",         type=int, default=8080)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--log_dir",      default="logs/so101_grpc")
    parser.add_argument("--num_rollouts", type=int, default=512)
    parser.add_argument("--num_elites",   type=int, default=64)
    parser.add_argument("--opt_iters",    type=int, default=5)
    parser.add_argument("--latent_dim",   type=int, default=512)
    parser.add_argument("--speed_scale",  type=float, default=1.0,
                        help="Scale max joint delta per step (0.0=frozen, 1.0=full speed). Default 1.0.")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Scale factor applied to discriminator reward (default: 1.0). "
                             "Reduces value/policy loss magnitude when raw reward is very large.")
    parser.add_argument("--lpf_alpha", type=float, default=0.6,
                        help="Low-pass filter weight on the new action (0=frozen, 1=no filter). "
                             "Applied every step in both RL and eval modes. Default 0.6.")
    parser.add_argument("--gripper_update_every", type=int, default=5,
                        help="Only allow a new gripper command every N action chunks. "
                             "Use 1 to update the gripper every chunk. Default 5.")
    parser.add_argument("--max_episode_steps", type=int, default=200,
                        help="Hard cap on transitions recorded per episode. "
                             "Steps beyond this are silently dropped. Default 200.")
    parser.add_argument("--load_checkpoint", default=None,
                        help="Path to a .pt checkpoint (e.g. logs/so101_grpc/models/model_ep40.pt) "
                             "to resume training from.")
    parser.add_argument("--eval", action="store_true",
                        help="Eval mode: load checkpoint, disable all training and exploration. "
                             "Requires --load_checkpoint.")
    parser.add_argument("--joint_dim",        type=int, default=512,
                        help="Output dim of the joint-state encoder stream (default: 1024 = 2× each camera stream).")
    parser.add_argument("--show_cameras", action="store_true",
                        help="Open a live OpenCV window showing both cameras side-by-side during training.")
    parser.add_argument("--no_wrist_cam", action="store_true",
                        help="Exclude wrist cam (cam) from encoder. Wrist cam looks down at gripper and "
                             "adds noise for directional tasks — use RealSense (cam2) only for scene info.")
    parser.add_argument("--replay_size", type=int, default=40_000,
                        help="Replay buffer capacity in steps (default: 40000 = 200 episodes × 200 steps).")
    parser.add_argument("--replay_batch_size", type=int, default=64,
                        help="Replay buffer batch size for training (default: 64). Reduce to save GPU memory.")
    parser.add_argument("--loss_horizon",    type=int, default=7,
                        help="Trajectory horizon for training loss AND MPPI planning (default: 7). "
                             "Must equal num_timesteps in PolicySamplingConfig — both are set together.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", default="so101-mpail2",
                        help="W&B project name (default: so101-mpail2).")
    parser.add_argument("--wandb_run_name", default=None,
                        help="W&B run name. Defaults to auto-generated name.")
    args = parser.parse_args()

    if not args.mock and args.demo_path is None:
        parser.error("--demo_path is required unless --mock is set.")
    if args.eval and not args.load_checkpoint:
        parser.error("--eval requires --load_checkpoint.")

    # Reduce CUDA memory fragmentation — helps when reserved-but-free > allocated
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    start_episode = 0

    # Demo recorder (optional)
    recorder = None
    if args.collect_dir:
        recorder = DemoRecorder(Path(args.collect_dir), flush_every=args.flush_every)
        logger.info(f"Demo recording enabled → {args.collect_dir}  (flush every {args.flush_every} steps)")

    # Runner (skipped in mock mode)
    runner = None
    if not args.mock:
        from mpail2.runner import MPAIL2Runner
        from mpail2.configs.cfgs import MPAIL2RunnerCfg, ObsNormalizerCfg
        from mpail2.configs.defs import (
            MultiCoderConfig, CNNCoderConfig, PlannerConfig, PolicySamplingConfig, LearnerConfig,
        )

        print(f"Loading demonstrations from {args.demo_path} ...")
        demonstrations = torch.load(args.demo_path, map_location="cpu", weights_only=False)
        demo_keys = {OBS_KEY, CAM_KEY, CAM2_KEY}
        demonstrations = {k: v.float() for k, v in demonstrations.items() if k in demo_keys}
        if CAM_KEY not in demonstrations:
            raise RuntimeError(f"Demo file missing '{CAM_KEY}'. Keys: {list(demonstrations)}")
        if CAM2_KEY not in demonstrations:
            logger.warning(f"Demo file missing '{CAM2_KEY}' — training without RealSense camera.")
            demonstrations[CAM2_KEY] = torch.zeros(
                demonstrations[CAM_KEY].shape[:-1] + (CAM2_C,),
                dtype=torch.float32
            )
        if OBS_KEY not in demonstrations:
            raise RuntimeError(f"Demo file missing '{OBS_KEY}'. Keys: {list(demonstrations)}")

        # Auto-trim state if demos contain follower+leader concatenated (e.g. 12D → 6D)
        state_demo_dim = demonstrations[OBS_KEY].shape[-1]
        if state_demo_dim != STATE_DIM:
            if state_demo_dim < STATE_DIM:
                raise RuntimeError(
                    f"Demo {OBS_KEY} has {state_demo_dim} dims but STATE_DIM={STATE_DIM}. "
                    f"Re-collect or re-convert demos."
                )
            demonstrations[OBS_KEY] = demonstrations[OBS_KEY][..., :STATE_DIM]
            print(f"  [auto-trim] {OBS_KEY}: {state_demo_dim}D → {STATE_DIM}D (follower joints only)")

        # Re-pair demo transitions to match online control frequency.
        # Demos collected at 30Hz have (obs_t, obs_{t+1}) pairs with ~33ms jumps.
        # Online training at 1Hz produces (obs_t, obs_{t+1}) pairs with ~1s jumps.
        # Without re-pairing the discriminator trivially separates demo (tiny EE changes)
        # from online (large EE changes) even at iteration 0 — the root cause of
        # reward separation before any training.
        # Fix: re-create pairs as (obs_at_t, obs_at_{t+K}) with K=demo_subsample.
        if args.demo_subsample > 1:
            K   = args.demo_subsample
            N_d = next(iter(demonstrations.values())).shape[0]
            n_new = N_d // K
            idxs_start = np.arange(n_new) * K
            idxs_end   = np.minimum(idxs_start + K - 1, N_d - 1)
            for key in list(demonstrations.keys()):
                t = demonstrations[key]          # [N_d, 2, ...]
                new_t = torch.stack([
                    t[torch.from_numpy(idxs_start), 0],   # obs  at t_start
                    t[torch.from_numpy(idxs_end),   1],   # nobs at t_start+K
                ], dim=1)
                demonstrations[key] = new_t
            print(f"  [re-pair ×{K}] {N_d} → {n_new} transitions  "
                  f"(demo {K}Hz steps ≈ online 1Hz steps)")

        # Convert demo joint states → EE proprioception
        print("  Converting demo joint states to EE proprioception (running FK)...")
        demo_joints = demonstrations[OBS_KEY].numpy()   # [N, 2, 6]
        N = demo_joints.shape[0]
        ee_proprio = np.zeros((N, 2, EE_PROPRIO_DIM), dtype=np.float32)
        for i in range(N):
            for j in range(2):
                ee_proprio[i, j] = _joints_to_ee_proprio(demo_joints[i, j])
        demonstrations[OBS_KEY] = torch.from_numpy(ee_proprio)
        print(f"  {OBS_KEY}: {tuple(demonstrations[OBS_KEY].shape)}  (EE proprio)")

        for k, v in demonstrations.items():
            print(f"  {k}: {tuple(v.shape)}")

        print("Building runner...")
        env = make_so101_env(SO101RealEnvArgs(device=device, mock=True,
                                              max_episode_length=args.max_episode_steps))
        _coder_list = [
            MultiCoderConfig.ProprioCoderConfig(obs_key=OBS_KEY, input_dim=EE_PROPRIO_DIM, output_dim=args.joint_dim),
            CNNCoderConfig(obs_key=CAM2_KEY, H=CAM2_H, W=CAM2_W, C=CAM2_C),  # RealSense — global view, arm position visible
        ]
        if not args.no_wrist_cam:
            _coder_list.append(
                CNNCoderConfig(obs_key=CAM_KEY, H=CAM_H, W=CAM_W, C=CAM_C),  # wrist cam — gripper state only
            )
        encoder_cfg = MultiCoderConfig(coder_list=_coder_list)
        planner_cfg = PlannerConfig(
            encoder_cfg=encoder_cfg,
            action_dim=ACTION_DIM,
            latent_dim=args.latent_dim,
            sampling_cfg=PolicySamplingConfig(
                num_rollouts=args.num_rollouts,
                num_timesteps=args.loss_horizon,  # must match loss_horizon
                policy_proportion=0.05,  # 5% policy rollouts, 95% random — same as Franka
                min_std=0.1,
                max_std=3.0,
            ),
            opt_iters=args.opt_iters,
            num_elites=args.num_elites,
        )
        learner_cfg = LearnerConfig(
            planner_cfg=planner_cfg,
            replay_size=args.replay_size,
            replay_batch_size=args.replay_batch_size,
            loss_horizon=args.loss_horizon,
            use_terminations=False,
            obs_normalizer_cfg=ObsNormalizerCfg(normalization_type="cam_only"),
        )
        if args.wandb:
            import dataclasses, wandb as _wandb

            def _cfg_to_dict(obj, prefix=""):
                """Recursively flatten a dataclass into a flat dict for wandb config."""
                out = {}
                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    for f in dataclasses.fields(obj):
                        val = getattr(obj, f.name)
                        key = f"{prefix}{f.name}"
                        if dataclasses.is_dataclass(val) and not isinstance(val, type):
                            out.update(_cfg_to_dict(val, prefix=f"{key}."))
                        elif isinstance(val, type):
                            pass  # skip class references
                        elif isinstance(val, dict):
                            out[key] = str(val)
                        elif isinstance(val, (int, float, str, bool, list, type(None))):
                            out[key] = val
                else:
                    out[prefix.rstrip(".")] = str(obj)
                return out

            wandb_cfg = {**vars(args)}
            wandb_cfg.update(_cfg_to_dict(learner_cfg, prefix="learner_cfg."))
            wandb_cfg.update(_cfg_to_dict(planner_cfg, prefix="planner_cfg."))

            _wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=wandb_cfg,
            )
            logger.info(f"W&B run: {_wandb.run.url}")

        log_cfg = MPAIL2RunnerCfg.LogCfg(
            log_dir=args.log_dir,
            checkpoint_every=999_999,
            no_wandb=not args.wandb,
            logger="wandb" if args.wandb else None,
            video_interval=999_999,
        )
        runner_cfg = MPAIL2RunnerCfg(
            learner_cfg=learner_cfg, log_cfg=log_cfg,
            num_learning_iterations=0, logger=None, vis_rollouts=False,
        )
        os.makedirs(os.path.join(args.log_dir, "models"), exist_ok=True)
        runner = MPAIL2Runner(demonstrations=demonstrations, env=env, runner_cfg=runner_cfg, device=device)
        start_episode = 0
        if args.load_checkpoint:
            print(f"Loading checkpoint from {args.load_checkpoint} ...")
            runner.load(args.load_checkpoint)
            _ckpt = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
            start_episode = int(_ckpt.get("iter", 0))
            print(f"  Checkpoint loaded. Resuming from episode {start_episode}.")

        # reward_scale is stored on the learner and applied only in value/policy updates,
        # NOT in update_reward() — scaling discriminator output breaks the GP Lipschitz constraint.
        runner.learner._reward_scale = float(args.reward_scale)
        print(f"  Reward scale: {args.reward_scale}")

        runner.learner.eval()
        print(f"Runner ready  device={device}  proprio_dim={EE_PROPRIO_DIM}  action_dim={ACTION_DIM}")
    else:
        logger.info("Mock mode — no runner loaded, returning home-position actions")

    # Start server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(
        MPAILServicer(
            runner=runner,
            device=device,
            mock=args.mock,
            recorder=recorder,
            speed_scale=args.speed_scale,
            eval_mode=args.eval,
            gripper_update_every=args.gripper_update_every,
            lpf_alpha=args.lpf_alpha,
            start_episode=start_episode,
            max_episode_steps=args.max_episode_steps,
            show_cameras=args.show_cameras,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    mode = "mock" if args.mock else ("eval" if args.eval else "rl")
    logger.info(f"gRPC server listening on port {args.port}  mode={mode}")
    logger.info("Waiting for robot_client.py ...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if recorder:
            recorder.flush()
            logger.info("Final demo flush complete.")
        server.stop(grace=2.0)


if __name__ == "__main__":
    serve()
