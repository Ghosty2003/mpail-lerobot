"""
policy_server.py — MPAIL2 online inference + learning server for the SO-101 robot.

LeRobot (separate conda env) sends observations and receives actions via HTTP.
The MPAIL learner stores every transition and periodically updates itself.

Run:
    conda activate mpail2
    python policy_server.py

LeRobot client side:
    Call POST /act  with {"obs": [float x 6]}           → get action
    Call POST /step_done  with full transition           → feed to learner
    Call POST /reset  when an episode ends               → reset planner state
"""

import torch
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from mpail2.runner import MPAIL2Runner
from mpail2.configs.cfgs import MPAIL2RunnerCfg
from mpail2.configs.defs import (
    MLPCoderConfig, PlannerConfig, PolicySamplingConfig, LearnerConfig,
)
from mpail2.envs.real.so101 import (
    SO101RealEnvArgs,
    make_so101_env,
    OBS_KEY,
    STATE_DIM,
    ACTION_DIM,
    HOME_POSITION_DEG,
    JOINT_LOWER_DEG,
    JOINT_UPPER_DEG,
    MAX_DELTA_DEG,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  –  edit to match your setup
# ─────────────────────────────────────────────────────────────────────────────
DEMO_PATH    = "raw_demos2_master.pt"   # converted from raw_demos2/ via convert.py
DEVICE       = "cuda"                   # "cpu" if no GPU
LATENT_DIM   = 512
NUM_ROLLOUTS = 512
NUM_ELITES   = 64
OPT_ITERS    = 5
MAX_EP_LEN   = 200                      # steps per episode
LOG_DIR      = "logs/so101_online"      # checkpoints written here

# SO-101 robot HTTP server (runs in LeRobot conda env)
ROBOT_HOST = "127.0.0.1"
ROBOT_PORT = 7070

# ─────────────────────────────────────────────────────────────────────────────
# BUILD ENV  (used only to initialise MPAIL2Runner dimensions; runner.learn()
# is never called here — actual obs/actions flow through the HTTP endpoints)
# ─────────────────────────────────────────────────────────────────────────────
print("Building SO-101 env (mock=True — no robot calls during init)...")
env = make_so101_env(SO101RealEnvArgs(
    device=DEVICE,
    host=ROBOT_HOST,
    port=ROBOT_PORT,
    max_episode_length=MAX_EP_LEN,
    mock=True,      # flip to False once the robot HTTP server is running
))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DEMONSTRATIONS
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading demonstrations from {DEMO_PATH} ...")
demonstrations = torch.load(DEMO_PATH, map_location="cpu", weights_only=False)
# Expected: {OBS_KEY: tensor(N, 2, STATE_DIM)}
# If the file is missing the obs key, remap it:
if OBS_KEY not in demonstrations:
    first_key = next(iter(demonstrations))
    print(f"[WARN] Key '{OBS_KEY}' not found — using '{first_key}' instead.")
    demonstrations = {OBS_KEY: demonstrations[first_key]}
demonstrations = {OBS_KEY: demonstrations[OBS_KEY].float()}
print(f"  {OBS_KEY}: {tuple(demonstrations[OBS_KEY].shape)}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD RUNNER
# ─────────────────────────────────────────────────────────────────────────────
print("Building MPAIL2 runner...")

encoder_cfg  = MLPCoderConfig(obs_key=OBS_KEY, input_dim=STATE_DIM, output_dim=LATENT_DIM)
sampling_cfg = PolicySamplingConfig(num_rollouts=NUM_ROLLOUTS)
planner_cfg  = PlannerConfig(
    encoder_cfg=encoder_cfg,
    action_dim=ACTION_DIM,
    latent_dim=LATENT_DIM,
    sampling_cfg=sampling_cfg,
    opt_iters=OPT_ITERS,
    num_elites=NUM_ELITES,
)
learner_cfg = LearnerConfig(
    planner_cfg=planner_cfg,
    replay_size=100_000,
    replay_batch_size=256,
    use_terminations=False,
)
log_cfg = MPAIL2RunnerCfg.LogCfg(
    log_dir=LOG_DIR, checkpoint_every=999_999, no_wandb=True, video_interval=999_999,
)
runner_cfg = MPAIL2RunnerCfg(
    learner_cfg=learner_cfg,
    log_cfg=log_cfg,
    num_learning_iterations=0,   # no offline loop; learning happens via /reset
    logger=None,
    vis_rollouts=False,
)

import os
os.makedirs(os.path.join(LOG_DIR, "models"), exist_ok=True)

runner = MPAIL2Runner(
    demonstrations=demonstrations,
    env=env,
    runner_cfg=runner_cfg,
    device=DEVICE,
)
runner.learner.eval()
print(f"Runner ready  obs_key={OBS_KEY}  obs_dim={STATE_DIM}  action_dim={ACTION_DIM}")

# ─────────────────────────────────────────────────────────────────────────────
# EPISODE / UPDATE TRACKING
# ─────────────────────────────────────────────────────────────────────────────
_episode_steps   = 0     # steps in the current episode (reset on /reset)
_episode_count   = 0     # total completed episodes
_total_updates   = 0     # total learner.update() calls
UPDATE_EVERY_EPS = 1     # run learner.update() every N completed episodes

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MPAIL2 SO-101 Policy Server")


class ObsRequest(BaseModel):
    obs: list[float]    # flat, length = STATE_DIM


class ActionResponse(BaseModel):
    action: list[float] # flat, length = ACTION_DIM


class TransitionRequest(BaseModel):
    obs:      list[float]
    action:   list[float]
    reward:   float
    next_obs: list[float]
    done:     bool


@app.get("/health")
def health():
    return {
        "status": "ok",
        "obs_key": OBS_KEY,
        "obs_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "episode_steps": _episode_steps,
        "episodes_completed": _episode_count,
        "total_updates": _total_updates,
        "replay_trajectories": runner.learner.storage.num_trajectories_stored,
    }


_HALF_RANGE = (JOINT_UPPER_DEG - JOINT_LOWER_DEG) / 2.0  # precomputed once


@app.post("/act", response_model=ActionResponse)
def act(req: ObsRequest):
    """Return a joint-position command in degrees for the given observation.

    Conversion:
        planner output (normalised [-1, 1])
        → absolute target  =  HOME + action * half_range
        → delta-clamped    (max MAX_DELTA_DEG per step from current obs)
        → clipped to joint limits
        → returned as degrees to the robot server
    """
    obs_t = torch.tensor(req.obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    obs_dict = {OBS_KEY: obs_t}
    with torch.no_grad():
        action_norm = runner.learner.act(obs_dict).squeeze(0).cpu().numpy()  # (ACTION_DIM,)

    # action=0 → home, action=±1 → home ± half_range
    target_deg = HOME_POSITION_DEG + action_norm * _HALF_RANGE

    # Speed limit: at most MAX_DELTA_DEG per step from where the robot currently is
    current_deg = np.array(req.obs, dtype=np.float32)
    delta = np.clip(target_deg - current_deg, -MAX_DELTA_DEG, MAX_DELTA_DEG)

    # Joint limit constraint: if already at a limit, block movement further into it
    delta[current_deg >= JOINT_UPPER_DEG] = np.minimum(
        delta[current_deg >= JOINT_UPPER_DEG], 0.0
    )
    delta[current_deg <= JOINT_LOWER_DEG] = np.maximum(
        delta[current_deg <= JOINT_LOWER_DEG], 0.0
    )

    action_deg = np.clip(current_deg + delta, JOINT_LOWER_DEG, JOINT_UPPER_DEG)

    return ActionResponse(action=action_deg.tolist())


def _parse_step_done_body(body: dict) -> tuple:
    """Extract (obs, next_obs, reward, done) from a /step_done body.

    Accepts two layouts:
      Flat:   {"obs": [...], "next_obs": [...], "reward": float, "done": bool}
      Nested: {"observation": {...joint keys...},
               "next_observation": {...},
               "reward": float, "done": bool}
    """
    def _joints_from_obs_dict(d: dict) -> list[float]:
        """Extract joint values from a planner_server-style observation dict."""
        vals = []
        for k, v in d.items():
            if k == "task":
                continue
            arr = np.array(v, dtype=np.float32)
            if arr.ndim < 3:          # scalar / 1-D  →  joint position
                vals.append(float(arr.flat[0]))
        return vals

    # --- obs ---
    if "obs" in body:
        obs = list(body["obs"])
    elif "observation" in body:
        obs = _joints_from_obs_dict(body["observation"])
    else:
        raise ValueError(f"Cannot find obs in body keys: {list(body.keys())}")

    # --- next_obs ---
    if "next_obs" in body:
        next_obs = list(body["next_obs"])
    elif "next_observation" in body:
        next_obs = _joints_from_obs_dict(body["next_observation"])
    else:
        next_obs = obs   # fallback: caller didn't send next_obs yet

    reward = float(body.get("reward", 0.0))
    done   = bool(body.get("done", False))
    return obs, next_obs, reward, done


@app.post("/step_done")
async def step_done(request: Request):
    """Feed one transition to the learner storage (no gradient update here — see /reset)."""
    global _episode_steps

    if _episode_steps >= MAX_EP_LEN:
        return {"status": "skipped", "reason": "buffer_full"}

    body = await request.json()
    try:
        obs_list, next_obs_list, reward_val, done_val = _parse_step_done_body(body)
    except Exception as exc:
        print(f"[step_done] parse error: {exc}\n  body keys: {list(body.keys())}")
        print(f"  full body: {body}")
        return {"status": "error", "detail": str(exc)}

    obs_dict      = {OBS_KEY: torch.tensor(obs_list,      dtype=torch.float32, device=DEVICE).unsqueeze(0)}
    next_obs_dict = {OBS_KEY: torch.tensor(next_obs_list, dtype=torch.float32, device=DEVICE).unsqueeze(0)}
    reward        = torch.tensor([reward_val], dtype=torch.float32, device=DEVICE)
    done          = torch.tensor([int(done_val)], dtype=torch.long, device=DEVICE)

    runner.learner.process_env_step(reward, done, {}, next_obs_dict)
    _episode_steps += 1

    with torch.no_grad():
        z  = runner.learner._encoder(obs_dict)
        zn = runner.learner._encoder(next_obs_dict)
        disc_reward = runner.learner._reward(z, zn).item()
    print(f"[step {_episode_steps}] disc_reward: {disc_reward:.4f}")

    return {"status": "ok", "episode_steps": _episode_steps}


@app.post("/debug/step_done")
async def debug_step_done(request: Request):
    """Call this once to print the exact body LeRobot sends to /step_done."""
    body = await request.json()
    print("\n=== /debug/step_done body ===")
    for k, v in body.items():
        if isinstance(v, (list, dict)):
            print(f"  {k!r}: type={type(v).__name__}  len={len(v)}")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"    {kk!r}: {vv!r}")
        else:
            print(f"  {k!r}: {v!r}")
    print("=============================\n")
    return {"status": "ok", "keys": list(body.keys())}


@app.post("/reset")
def reset_episode():
    """End of episode: run one learner update (uses the full episode just collected),
    then reset planner state and rollout buffer for the next episode."""
    global _episode_steps, _episode_count, _total_updates

    _episode_count += 1
    updated = False

    # Only update when there is replay data (requires at least one prior episode)
    if runner.learner.storage.has_replay_data:
        if _episode_count % UPDATE_EVERY_EPS == 0:
            runner.learner.train()
            train_stats = runner.learner.update(iteration=_episode_count)
            runner.learner.eval()
            _total_updates += 1
            updated = True
            print(
                f"[ep {_episode_count}] "
                f"dyn={train_stats.get('Dyn/mean_loss', 0):.4f}  "
                f"reward={train_stats.get('Reward/mean_gen_reward', 0):.4f}  "
                f"value={train_stats.get('Value/mean_loss', 0):.4f}  "
                f"policy={train_stats.get('Policy/mean_loss', 0):.4f}"
            )

    runner.learner.planner.reset()
    runner.learner.storage.clear()
    _episode_steps = 0

    # Periodic checkpoint
    if _episode_count % 20 == 0:
        runner.save(postfix=f"ep{_episode_count}")

    return {
        "status": "ok",
        "episode": _episode_count,
        "updated": updated,
        "total_updates": _total_updates,
    }


@app.post("/save")
def save_checkpoint():
    """Manually save the current model weights to LOG_DIR/models/."""
    runner.save(postfix=f"ep{_episode_count}_manual")
    path = os.path.join(LOG_DIR, "models", f"model_ep{_episode_count}_manual.pt")
    return {"status": "saved", "path": path}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
