"""
planner_server.py — HTTP inference + data collection server backed by the mpail2 Planner.

Features:
  1. Load the mpail2 Planner (encoder + dynamics + reward + value + policy network)
     from a checkpoint produced by MPAIL2Runner.save().
  2. Receive observations from ExternalPlannerServer via HTTP POST /plan.
  3. Run MPPI planning with Planner.act() and return the chosen action to the robot.
  4. Optionally record every (obs, action) transition to disk for future training.

Usage:
    python planner_server.py \
        --checkpoint logs/models/model_5000.pt \
        --config_pkl logs/planner_cfg.pkl \
        --host 0.0.0.0 \
        --port 9090 \
        --device cuda \
        --obs_key observation.state \
        --collect_dir ./collected_data

Start ExternalPlannerServer alongside:
    python -m lerobot.async_inference.external_planner_server \
        --port 8080 \
        --planner_url http://localhost:9090/plan

Serialise PlannerCfg for loading (run once after training):
    import pickle
    from your_train_config import cfg  # your GymHydraTrainConfig or equivalent
    with open("logs/planner_cfg.pkl", "wb") as f:
        pickle.dump(cfg.runner.learner_cfg.planner_cfg, f)
"""

import argparse
import logging
import pickle
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("planner_server")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class ServerState:
    planner: Any = None          # mpail2.planner.Planner instance; None in mock mode
    mock: bool = False           # When True, skip Planner and return zero actions
    mock_action_dim: int = 6     # Action dimension used in mock mode (SO-101 default: 6)
    device: str = "cpu"
    obs_key: str = "observation.state"  # Key used to store the joint state tensor for the Planner
    collect_dir: Optional[Path] = None  # Directory for saving collected data; None = disabled
    flush_every: int = 200              # Flush buffer to disk every N steps

    # Data buffer (thread-safe via _lock)
    _lock = threading.Lock()
    _obs_buf: List[np.ndarray] = []
    _next_obs_buf: List[np.ndarray] = []
    _action_buf: List[np.ndarray] = []
    _pending_obs: Optional[np.ndarray] = None  # Previous obs waiting to be paired with next obs
    _file_idx: int = 0


state = ServerState()


# ---------------------------------------------------------------------------
# Planner loading
# ---------------------------------------------------------------------------

def load_planner(checkpoint_path: str, config_pkl_path: str, device: str):
    """
    Reconstruct the mpail2 Planner from a checkpoint and load its weights.

    Checkpoint structure (written by MPAIL2Runner.save()):
        {
            "model_state_dict": planner.state_dict(),
            "reward_optimizer_state_dict": ...,
            "value_optimizer_state_dict": ...,
            "iter": int,
        }

    config_pkl must contain a serialised PlannerCfg dataclass object.
    """
    from mpail2.planner import Planner

    logger.info(f"Loading PlannerCfg from {config_pkl_path}")
    with open(config_pkl_path, "rb") as f:
        planner_cfg = pickle.load(f)  # nosec

    logger.info(f"Building Planner (num_envs=1, device={device})")
    planner = Planner(
        policy_config=planner_cfg,
        num_envs=1,
        device=device,
        dtype=torch.float32,
    )

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    planner.load_state_dict(ckpt["model_state_dict"])
    planner.eval()

    trained_iter = ckpt.get("iter", "unknown")
    logger.info(f"Planner loaded. Trained for {trained_iter} iterations.")
    return planner


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------

def _parse_observation(observation: dict) -> tuple[np.ndarray, dict]:
    """
    Parse the observation dict sent by ExternalPlannerServer.

    Individual joint position keys (order matches robot.action_features):
        "shoulder_pan.pos":  float
        "shoulder_lift.pos": float
        "elbow_flex.pos":    float
        "wrist_flex.pos":    float
        "wrist_roll.pos":    float
        "gripper.pos":       float

    Camera images use the camera name as key, value is a nested list (H, W, 3) uint8:
        "front": [[[r,g,b], ...], ...]

    "task" is a string and is ignored here.

    Returns:
        joint_state_np  — shape (state_dim,) float32, joint positions concatenated in key order
        image_arrays    — {camera_name: np.ndarray (H, W, C) uint8}
    """
    joint_vals: List[float] = []
    image_arrays: dict = {}

    for key, value in observation.items():
        if key == "task":
            continue

        arr = np.array(value, dtype=np.float32)

        if arr.ndim == 3:
            # Image key (e.g. "front")
            image_arrays[key] = arr
        else:
            # Scalar joint position (ndim == 0 or ndim == 1 with length 1)
            joint_vals.append(float(arr.flat[0]))

    joint_state_np = np.array(joint_vals, dtype=np.float32)
    return joint_state_np, image_arrays


def _json_obs_to_planner_input(
    observation: dict,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Convert a parsed observation into the dict[str, Tensor] format expected by Planner.act().

    Joint positions are concatenated into a single vector stored under state.obs_key
    (default "observation.state"). encoder_cfg.obs_key must match state.obs_key.

    Camera images (optional) are converted to (1, C, H, W) float32 in [0, 1].
    """
    joint_state_np, image_arrays = _parse_observation(observation)

    result: Dict[str, torch.Tensor] = {}

    # Joint state vector -> (1, state_dim)
    result[state.obs_key] = (
        torch.from_numpy(joint_state_np).unsqueeze(0).to(device)
    )

    # Images (optional) -> (1, C, H, W) normalised to [0, 1]
    for cam_name, img_arr in image_arrays.items():
        img = img_arr.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # (C, H, W)
        result[cam_name] = torch.from_numpy(img).unsqueeze(0).to(device)

    return result


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _record_step(obs_arr: np.ndarray, action_arr: np.ndarray):
    """
    Append one transition to the buffer.

    (obs[t], obs[t+1]) pairs are formed with a one-step lag so that next_obs
    is available before writing. Flushes to disk when the buffer is full.
    """
    if state.collect_dir is None:
        return

    with state._lock:
        if state._pending_obs is not None:
            state._obs_buf.append(state._pending_obs)
            state._next_obs_buf.append(obs_arr.copy())
            state._action_buf.append(action_arr.copy())

        state._pending_obs = obs_arr.copy()

        if len(state._obs_buf) >= state.flush_every:
            _flush_to_disk()


def _flush_to_disk():
    """
    Write the buffer to a compressed numpy file compatible with DictDemoDataset:
        {obs_key: array of shape (N, 2, *obs_shape)}  (index 0 = obs, index 1 = next_obs)

    Caller must hold _lock.
    """
    if not state._obs_buf:
        return

    obs_arr = np.stack(state._obs_buf, axis=0)           # (N, state_dim)
    next_obs_arr = np.stack(state._next_obs_buf, axis=0)
    actions_arr = np.stack(state._action_buf, axis=0)    # (N, action_dim)

    paired = np.stack([obs_arr, next_obs_arr], axis=1)   # (N, 2, state_dim)

    save_path = state.collect_dir / f"traj_{state._file_idx:04d}.npz"
    np.savez_compressed(
        save_path,
        **{state.obs_key: paired},
        actions=actions_arr,
    )
    logger.info(f"Saved {len(state._obs_buf)} steps to {save_path}")

    state._obs_buf.clear()
    state._next_obs_buf.clear()
    state._action_buf.clear()
    state._file_idx += 1


def flush_remaining():
    """Flush any remaining buffered transitions to disk before shutdown."""
    with state._lock:
        _flush_to_disk()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.collect_dir is not None:
        logger.info("Server shutting down — flushing remaining data...")
        flush_remaining()
        logger.info("Data flush complete.")


app = FastAPI(title="mpail2 Planner Server", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check — returns server and planner status."""
    return {
        "status": "ok",
        "mode": "mock (zero actions)" if state.mock else "planner (live inference)",
        "planner_loaded": state.planner is not None,
        "mock_action_dim": state.mock_action_dim if state.mock else None,
        "device": state.device,
        "collect_dir": str(state.collect_dir) if state.collect_dir else None,
        "buffered_steps": len(state._obs_buf),
        "saved_files": state._file_idx,
    }


@app.post("/plan")
async def plan(request: Request):
    """
    Receive an observation from ExternalPlannerServer, run MPPI planning, and
    return the resulting action chunk.

    Request body (JSON):
    {
        "timestep": int,
        "timestamp": float,
        "must_go": bool,
        "observation": {
            "task": "string",              // optional, ignored
            "shoulder_pan.pos":  float,    // one key per joint, in action_features order
            "shoulder_lift.pos": float,
            "elbow_flex.pos":    float,
            "wrist_flex.pos":    float,
            "wrist_roll.pos":    float,
            "gripper.pos":       float,
            "front": [...],                // optional camera image (H, W, 3) uint8
        }
    }

    Response body (JSON):
    {
        "actions": [
            [float, ...]   // one action vector per timestep; tanh-squashed to [-1, 1]
        ]
    }
    """
    if not state.mock and state.planner is None:
        raise HTTPException(status_code=503, detail="Planner not loaded and mock mode is off.")

    body = await request.json()
    timestep: int = body.get("timestep", 0)
    must_go: bool = body.get("must_go", False)
    observation: dict = body.get("observation", {})

    if not observation:
        raise HTTPException(status_code=400, detail="observation field is empty.")

    logger.info(
        f"Received observation timestep={timestep} must_go={must_go} "
        f"keys={list(observation.keys())}"
    )

    t0 = time.perf_counter()

    if state.mock:
        action_np = np.zeros(state.mock_action_dim, dtype=np.float32)
        logger.info(f"[MOCK] Returning zero action dim={state.mock_action_dim}")
    else:
        try:
            obs_input = _json_obs_to_planner_input(observation, state.device)
        except Exception as exc:
            logger.exception("Failed to parse observation.")
            raise HTTPException(status_code=400, detail=f"Observation parse error: {exc}")

        try:
            with torch.inference_mode():
                # Planner.act() returns [num_envs=1, action_dim]
                action_tensor = state.planner.act(obs_input)
        except Exception as exc:
            logger.exception("Planner inference failed.")
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

        # Squeeze batch dim: (1, action_dim) -> (action_dim,)
        action_np = action_tensor.squeeze(0).detach().cpu().numpy()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Action computed in {elapsed_ms:.1f}ms: {action_np.tolist()}")

    # Record transition (joint state vector concatenated from per-key scalars)
    obs_np, _ = _parse_observation(observation)
    _record_step(obs_arr=obs_np, action_arr=action_np)

    # ExternalPlannerServer expects a list of action vectors
    return JSONResponse(content={"actions": [action_np.tolist()]})


# ---------------------------------------------------------------------------
# Manual flush endpoint
# ---------------------------------------------------------------------------

@app.post("/flush")
async def flush():
    """Manually flush the data buffer to disk without restarting the server."""
    flush_remaining()
    return {"status": "flushed", "saved_files": state._file_idx}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="mpail2 HTTP inference + data collection server")
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock mode: skip Planner loading and return zero actions. Useful for testing the communication pipeline.",
    )
    parser.add_argument(
        "--mock_action_dim", type=int, default=6,
        help="Action dimension returned in mock mode (SO-101 default: 6).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to mpail2 checkpoint, e.g. logs/models/model_5000.pt. Not required in mock mode.",
    )
    parser.add_argument(
        "--config_pkl", default=None,
        help="Path to serialised PlannerCfg pickle. Not required in mock mode.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=9090,
        help="Bind port (default: 9090).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Inference device: cpu / cuda / mps / auto (default: auto).",
    )
    parser.add_argument(
        "--obs_key", default="observation.state",
        help=(
            "Key used to store the concatenated joint state tensor passed to the Planner. "
            "Must match PlannerCfg.encoder_cfg.obs_key (default: 'observation.state')."
        ),
    )
    parser.add_argument(
        "--collect_dir", default=None,
        help="Directory for saving collected (obs, action) data. Omit to disable collection.",
    )
    parser.add_argument(
        "--flush_every", type=int, default=200,
        help="Write buffer to disk every N steps (default: 200).",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = args.device

    # Data collection setup
    if args.collect_dir is not None:
        collect_path = Path(args.collect_dir)
        collect_path.mkdir(parents=True, exist_ok=True)
        state.collect_dir = collect_path
        logger.info(f"Data collection enabled. Saving to: {collect_path}")
    else:
        logger.info("Data collection disabled (--collect_dir not set).")

    state.obs_key = args.obs_key
    state.flush_every = args.flush_every
    state.device = resolved_device
    state.mock = args.mock
    state.mock_action_dim = args.mock_action_dim

    if args.mock:
        logger.info(
            f"[MOCK MODE] Skipping Planner load. "
            f"Will return zero actions with dim={args.mock_action_dim}."
        )
    else:
        if not args.checkpoint or not args.config_pkl:
            parser.error("--checkpoint and --config_pkl are required when not using --mock.")
        state.planner = load_planner(
            checkpoint_path=args.checkpoint,
            config_pkl_path=args.config_pkl,
            device=resolved_device,
        )

    logger.info(
        f"Server starting on {args.host}:{args.port} | "
        f"mode={'mock' if args.mock else 'planner'} | device={resolved_device}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
