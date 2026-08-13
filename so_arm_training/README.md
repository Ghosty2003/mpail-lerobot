# so_arm_training

Everything needed to record demonstrations on the real SO-101 arm and train
MPAIL2 on them: the gRPC servers that talk to the robot, the demo-collection
client patch for `lerobot`, and the conversion/training/diagnostic scripts.
See [`docs/SO101_GETTING_STARTED.md`](../docs/SO101_GETTING_STARTED.md) for
the full install walkthrough — this file focuses on how the pieces in this
folder fit together and how to run them.

## Running scripts in this folder

Always invoke scripts here as a module, from the **repo root** — not
`python so_arm_training/foo.py`:

```bash
python -m so_arm_training.demo_recording_server --collect_dir ./raw_demos2
python -m so_arm_training.train_so101_local --demo_path demo.pt ...
```

Why: several of these scripts import `ik_utils` and `transport` (plain
top-level modules at the repo root, not part of the installed `mpail2`
package). Running a script directly as `python so_arm_training/foo.py` puts
only `so_arm_training/` on `sys.path`, so those imports fail. Running it as
`python -m so_arm_training.foo` puts the current directory (the repo root, if
that's where you run it from) on `sys.path` instead, so `ik_utils`/`transport`
resolve correctly. `mpail2` itself is unaffected either way since it's
installed editable (`pip install -e .`).

## Two conda envs

| Env | Used for |
| --- | --- |
| `lerobot` | `so101_robot_server.py` (owns the physical arm + cameras), `convert_lerobot.py`, the `lerobot.async_inference.robot_client` teleop client |
| `mpail2` | `demo_recording_server.py`, `train_so101_local.py`, `convert.py`, `replay_demo.py`, `check_encoder_collapse.py` — anything importing the `mpail2` package |

## Deploying the async_inference patch

`async_inference/` in this folder is a patched copy of
`lerobot/src/lerobot/async_inference/` — it fixes a camera-reconnect bug, an
off-by-one in the timestep counter, and adds a block-until-server-ready sync
point before homing between episodes/training updates. Deploy it into your
`lerobot` clone once (re-run after pulling upstream `lerobot` changes, or
after editing anything under `async_inference/` here):

```bash
cp -r so_arm_training/async_inference/* <path-to-lerobot-clone>/src/lerobot/async_inference/
```

## Workflow

### 1. Start the robot-side server (`lerobot` env)

```bash
conda activate lerobot
python -m so_arm_training.so101_robot_server \
    --robot_port /dev/ttyACM0 --robot_id Kid \
    --cam_index /dev/video0 --cam2_serial 317422074482 --grpc_port 7070
```

Owns the arm + both cameras. Used by `train_so101_local.py` (online training
loop). Demo *collection* below uses a separate, simpler server
(`demo_recording_server.py`) instead — don't run both against the same
hardware at once.

### 2. Collect demonstrations

Start the recording server (`mpail2` env):

```bash
conda activate mpail2
python -m so_arm_training.demo_recording_server --collect_dir ./raw_demos2 --flush_every 200
```

It holds the arm at its current position (or drives home between episodes)
and saves every `(obs_t, obs_t+1)` pair it receives to `raw_demos2/traj_*.npz`.
It doesn't move the arm itself — that's the leader arm's job, via the
standard `lerobot` teleop client (`lerobot` env, separate terminal):

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=Kid \
    --robot.cameras="{cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, cam2: {type: intelrealsense, serial_number_or_name: 317422074482, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=Mom \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=dummy \
    --actions_per_chunk=1 \
    --task="pick up the cup"
```

Flag notes:
- `--robot.port` / `--teleop.port`: follower / leader arm serial ports.
- `--robot.cameras`: both cams in one JSON-ish dict — `cam` (wrist, OpenCV) and
  `cam2` (RealSense, by serial number).
- `--server_address`: must match `demo_recording_server.py`'s `--port` (default 8080).
- `--policy_type` / `--pretrained_name_or_path`: required by `robot_client`'s
  CLI even in this recording setup, where the server ignores them and just
  echoes joint state back — `act` / `dummy` are placeholders, not a real policy.
- With the leader connected, physically move it to demonstrate the task; the
  follower mirrors it and every step gets recorded. Each episode auto-ends
  (saves, homes, pauses, resumes) after `--max_episode_steps` (default 200)
  steps on the server side.

### 3. Convert to training format

```bash
conda activate mpail2
python -m so_arm_training.convert --dirs raw_demos2 --out demo.pt --img_w 64 --img_h 48
```

(Recorded via a LeRobot dataset instead of `demo_recording_server.py`? Use
`convert_lerobot.py` in the `lerobot` env instead — see its own docstring.)

Sanity-check a trajectory by replaying it on the real arm:

```bash
conda activate lerobot
python -m so_arm_training.replay_demo raw_demos2/traj_0000.npz --port /dev/ttyACM0 --robot_id Kid
```

### 4. Train

With `so101_robot_server.py` (step 1) still running:

```bash
conda activate mpail2
python -m so_arm_training.train_so101_local \
    --demo_path demo.pt --robot_host 127.0.0.1 --robot_port 7070 \
    --device cuda --speed_scale 0.4 --lpf_alpha 0.5 --wandb
```

See `train_so101_local.py --help` for the full flag list (MPPI sampling,
gripper hold steps, checkpointing, eval mode, ...).

### 5. Diagnose encoder collapse (optional)

```bash
conda activate mpail2
python -m so_arm_training.check_encoder_collapse --demo_path demo.pt \
    --checkpoint logs/so101_local/models/model_N.pt
```

Reports per-dimension latent std and effective rank (participation ratio) —
a low effective rank relative to `latent_dim` is the signature of
representation collapse even when the JEP/dynamics loss looks fine.

## File reference

| File | Env | Purpose |
| --- | --- | --- |
| `so101_robot_server.py` | `lerobot` | gRPC server owning the physical arm + cameras; backs `train_so101_local.py`'s online RL loop |
| `demo_recording_server.py` | `mpail2` | gRPC server that just records `(obs, next_obs)` pairs to `.npz` while you teleoperate |
| `async_inference/` | — | Patched copy of `lerobot/src/lerobot/async_inference/`; deploy into your `lerobot` clone |
| `convert.py` | `mpail2` | `raw_demos*/*.npz` → single `demo.pt` |
| `convert_lerobot.py` | `lerobot` | LeRobot-recorded dataset → `demo.pt` |
| `train_so101_local.py` | `mpail2` | Main MPAIL2 training entry point (in-process env loop, no dropped observations) |
| `replay_demo.py` | `lerobot` | Replay a recorded `.npz` trajectory on the real arm, or preview its camera frames, as a sanity check |
| `check_encoder_collapse.py` | `mpail2` | Offline diagnostic: encoder latent effective-rank / collapse check against a checkpoint |
