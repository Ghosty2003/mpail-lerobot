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
        --max_episode_steps=100 \
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
    SO101RealEnvArgs, make_so101_env, OBS_KEY, STATE_DIM, ACTION_DIM,
    HOME_POSITION_DEG, JOINT_LOWER_DEG, JOINT_UPPER_DEG, MAX_DELTA_DEG,
)
from mpail2.envs.real.so101.robot_limits import CAM_KEY, CAM_H, CAM_W, CAM_C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("grpc_policy_server")

_HALF_RANGE = (JOINT_UPPER_DEG - JOINT_LOWER_DEG) / 2.0


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
        if key == "task":
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


def _build_obs_dict(joint_state: np.ndarray, image_arrays: dict, device: str) -> dict:
    result = {OBS_KEY: torch.from_numpy(joint_state).unsqueeze(0).to(device)}
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


def _to_degrees(action_norm: np.ndarray, current_joints: np.ndarray, speed_scale: float = 1.0) -> np.ndarray:
    max_delta = MAX_DELTA_DEG * speed_scale
    target = HOME_POSITION_DEG + action_norm * _HALF_RANGE
    delta  = np.clip(target - current_joints, -max_delta, max_delta)
    delta[current_joints >= JOINT_UPPER_DEG] = np.minimum(delta[current_joints >= JOINT_UPPER_DEG], 0.0)
    delta[current_joints <= JOINT_LOWER_DEG] = np.maximum(delta[current_joints <= JOINT_LOWER_DEG], 0.0)
    return np.clip(current_joints + delta, JOINT_LOWER_DEG, JOINT_UPPER_DEG).astype(np.float32)


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

    def record(self, obs: np.ndarray, images: Dict[str, np.ndarray]):
        with self._lock:
            if self._pending_obs is not None:
                self._obs_buf.append(self._pending_obs)
                self._next_obs_buf.append(obs.copy())
                for cam, img in self._pending_images.items():
                    self._image_bufs.setdefault(cam, []).append(img)
                    self._image_next_bufs.setdefault(cam, []).append(
                        images.get(cam, img).copy()
                    )

            self._pending_obs    = obs.copy()
            self._pending_images = {k: v.copy() for k, v in images.items()}

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
        np.savez_compressed(str(path), **save_data)
        n = len(self._obs_buf)
        cams = list(self._image_bufs.keys())
        logger.info(f"[recorder] Saved {n} steps → {path}  cameras={cams or 'none'}")

        self._obs_buf.clear(); self._next_obs_buf.clear()
        self._image_bufs.clear(); self._image_next_bufs.clear()
        self._file_idx += 1


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
    ):
        self.runner      = runner      # None in mock mode
        self.device      = device
        self.mock        = mock
        self.eval_mode   = eval_mode
        self.recorder    = recorder
        self.speed_scale = speed_scale
        self.gripper_update_every = max(1, gripper_update_every)

        self._lock = threading.Lock()
        self._obs_queue: Queue = Queue(maxsize=1)
        self.shutdown_event = threading.Event()

        self._prev_obs_dict  = None
        self._transition_open = False  # set after learner.act(), cleared after process_env_step
        self._episode_steps  = 0
        self._episode_count  = 0
        self._prev_timestep  = -1
        self._recording_paused = False   # True while waiting for Enter between episodes
        self._actions_per_chunk = 1      # updated from SendPolicyInstructions

        # Exploration is handled by MPPI sampling (policy_proportion=0.05, max_std=2.0)
        # — same as Franka. No episode-level noise decay needed.
        self._prev_action_norm = None
        self._lpf_alpha        = 0.6   # EMA weight on new action (lower = smoother, more lag)
        self._last_gripper_deg = None
        self._gripper_chunk_count = 0

    def Ready(self, request, context):
        # Train synchronously on the just-completed episode.
        # The client blocks on this call, so no observations arrive during training.
        if not self.mock and not self.eval_mode and self._episode_steps > 0:
            logger.info(f"Ready: training on episode {self._episode_count + 1} ({self._episode_steps} steps)...")
            self._on_episode_end()
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
            logger.info(f"SendObservations: received {n_chunks} chunk(s), {len(data)} bytes")
        except Exception as e:
            logger.exception(f"SendObservations chunk read error: {e}")
            return services_pb2.Empty()

        try:
            timed_obs: TimedObservation = _loads(data)
        except Exception as e:
            logger.exception(f"SendObservations deserialize error: {e}  data[:64]={data[:64]!r}")
            return services_pb2.Empty()

        try:
            timestep = timed_obs.get_timestep()
            joint_state, image_arrays = _parse_raw_obs(timed_obs.get_observation())

            self._prev_timestep = timestep

            # Record demo transition (skipped while paused between episodes)
            if self.recorder is not None and not self._recording_paused:
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
                    buffer_full = self._episode_steps >= self.runner.learner.storage.num_steps_per_env
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
                    logger.info(
                        f"[ep {self._episode_count + 1} step {self._episode_steps}]"
                        f"  disc_reward={disc_r:.4f}"
                    )

                with torch.no_grad():
                    self.runner.learner.act(obs_dict)   # runs MPPI, fills _opt_controls
                self._prev_obs_dict = obs_dict
                self._transition_open = not self.eval_mode

            # Full planned trajectory: (num_timesteps, action_dim)
            traj_norm = self.runner.learner.planner._opt_controls[0].detach().cpu().numpy()

            # Build action chunk — propagate joint positions so delta-clamping is consistent
            timed_actions = []
            current = joint_state.copy()
            gripper_index = ACTION_DIM - 1
            update_gripper = (
                self._last_gripper_deg is None
                or self._gripper_chunk_count % self.gripper_update_every == 0
            )
            for t_offset, step_norm in enumerate(traj_norm):
                action_norm = step_norm.copy()
                if self.eval_mode and self._prev_action_norm is not None:
                    action_norm = self._lpf_alpha * action_norm + (1 - self._lpf_alpha) * self._prev_action_norm
                    action_norm = np.clip(action_norm, -1.0, 1.0)
                if self.eval_mode:
                    self._prev_action_norm = action_norm.copy()
                step_deg = _to_degrees(action_norm, current, self.speed_scale)
                if update_gripper:
                    self._last_gripper_deg = float(step_deg[gripper_index])
                else:
                    step_deg[gripper_index] = self._last_gripper_deg
                timed_actions.append(TimedAction(
                    timestamp=timestamp,
                    timestep=timestep + t_offset,
                    action=torch.tensor(step_deg, dtype=torch.float32),
                ))
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
                self.recorder.flush()
                self._recording_paused = True
                print(
                    f"\n── Episode {self._episode_count} saved ──────────────────────────\n"
                    f"   Press Enter to start recording episode {self._episode_count + 1} "
                    f"(Ctrl-C to stop)...",
                    flush=True,
                )
                threading.Thread(target=self._wait_for_enter, daemon=True).start()
            return

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
            self.runner.learner.planner.reset()
            self.runner.learner.storage.clear()

        self._episode_steps = 0
        self._prev_obs_dict = None
        self._transition_open = False
        self._prev_action_norm = None
        self._last_gripper_deg = None
        self._gripper_chunk_count = 0

        logger.info(f"[ep {self._episode_count}] training done")

        if train_stats:
            logger.info(
                f"[ep {self._episode_count}] "
                f"dyn={train_stats.get('Dyn/mean_loss', 0):.4f}  "
                f"reward={train_stats.get('Reward/mean_gen_reward', 0):.4f}  "
                f"value={train_stats.get('Value/mean_loss', 0):.4f}  "
                f"policy={train_stats.get('Policy/mean_loss', 0):.4f}"
            )
        if self._episode_count % 20 == 0:
            self.runner.save(postfix=f"ep{self._episode_count}")

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
    parser.add_argument("--mock",         action="store_true",
                        help="Skip runner; return home-position actions. Use with --collect_dir to record demos.")
    parser.add_argument("--collect_dir",  default=None,
                        help="Directory to save recorded (obs, next_obs) .npz files.")
    parser.add_argument("--flush_every",  type=int, default=200,
                        help="Flush buffer to disk every N steps (default: 200).")
    parser.add_argument("--port",         type=int, default=8080)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--log_dir",      default="logs/so101_grpc")
    parser.add_argument("--num_rollouts", type=int, default=512)
    parser.add_argument("--num_elites",   type=int, default=64)
    parser.add_argument("--opt_iters",    type=int, default=5)
    parser.add_argument("--latent_dim",   type=int, default=512)
    parser.add_argument("--speed_scale",  type=float, default=0.3,
                        help="Scale max joint delta per step (0.0=frozen, 1.0=full speed). Default 0.3.")
    parser.add_argument("--gripper_update_every", type=int, default=5,
                        help="Only allow a new gripper command every N action chunks. "
                             "Use 1 to update the gripper every chunk. Default 5.")
    parser.add_argument("--load_checkpoint", default=None,
                        help="Path to a .pt checkpoint (e.g. logs/so101_grpc/models/model_ep40.pt) "
                             "to resume training from.")
    parser.add_argument("--eval", action="store_true",
                        help="Eval mode: load checkpoint, disable all training and exploration. "
                             "Requires --load_checkpoint.")
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

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

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
        demo_keys = {OBS_KEY, CAM_KEY}
        demonstrations = {k: v.float() for k, v in demonstrations.items() if k in demo_keys}
        if CAM_KEY not in demonstrations:
            raise RuntimeError(f"Demo file missing '{CAM_KEY}'. Keys: {list(demonstrations)}")
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

        for k, v in demonstrations.items():
            print(f"  {k}: {tuple(v.shape)}")

        print("Building runner...")
        env = make_so101_env(SO101RealEnvArgs(device=device, mock=True))
        encoder_cfg = MultiCoderConfig(coder_list=[
            MultiCoderConfig.ProprioCoderConfig(obs_key=OBS_KEY, input_dim=STATE_DIM, output_dim=64),
            CNNCoderConfig(obs_key=CAM_KEY, H=CAM_H, W=CAM_W, C=CAM_C),
        ])
        planner_cfg = PlannerConfig(
            encoder_cfg=encoder_cfg,
            action_dim=ACTION_DIM,
            latent_dim=args.latent_dim,
            sampling_cfg=PolicySamplingConfig(
                num_rollouts=args.num_rollouts,
                policy_proportion=0.05,  # 5% policy rollouts, 95% random — same as Franka
                min_std=0.05,
                max_std=2.0,
            ),
            opt_iters=args.opt_iters,
            num_elites=args.num_elites,
        )
        learner_cfg = LearnerConfig(
            planner_cfg=planner_cfg,
            replay_size=10_000,
            replay_batch_size=256,
            use_terminations=False,
            obs_normalizer_cfg=ObsNormalizerCfg(normalization_type="fixed"),
        )
        if args.wandb:
            import wandb as _wandb
            _wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
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
        if args.load_checkpoint:
            print(f"Loading checkpoint from {args.load_checkpoint} ...")
            runner.load(args.load_checkpoint)
            print("  Checkpoint loaded.")
        runner.learner.eval()
        print(f"Runner ready  device={device}  obs_dim={STATE_DIM}  action_dim={ACTION_DIM}")
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
