# Getting Started: MPAIL2 on the Real SO-101 Arm

This is a command-by-command walkthrough for someone new to this repo who wants
to get the real-robot (SO-101) MPAIL2 pipeline running end to end: environment
setup → hardware calibration → demo collection → training → eval.

It complements the top-level [README.md](../README.md) / [INSTALL.md](INSTALL.md),
which cover the general (Isaac/Gym) install. This doc is specific to the SO-101
real-robot stack added on top of that.

## 0. Repo layout (what lives where)

This workflow spans **two repos**, each with its own conda env:

| Repo | Env | Python | Role |
| --- | --- | --- | --- |
| `mpail-lerobot` (this repo) | `mpail2` | 3.10 | MPAIL2 algorithm: encoder/dynamics/reward/value/policy, training loop, gRPC *client* side |
| [`lerobot`](https://github.com/huggingface/lerobot) (HuggingFace fork, cloned separately) | `lerobot` | 3.12 | Robot/camera drivers (Feetech servos, OpenCV/RealSense cameras), gRPC *server* side, calibration/teleop CLIs |

They talk to each other over local gRPC — `so101_robot_server.py` (runs in the
`lerobot` env, owns the physical arm + cameras) serves requests from either
`train_so101_local.py` or `demo_recording_server.py` (run in the `mpail2` env).

## 1. Install

### 1a. This repo (`mpail2` env)

```bash
git clone <this-repo-url> mpail-lerobot
cd mpail-lerobot
conda create -n mpail2 python=3.10
conda activate mpail2
pip install --upgrade pip
pip install -e .
pip install grpcio grpcio-tools   # not in pyproject.toml; needed for the transport/ gRPC stubs
```

### 1b. `lerobot` (separate repo + env)

```bash
git clone https://github.com/huggingface/lerobot.git ~/Desktop/lerobot
conda create -n lerobot python=3.12
conda activate lerobot
cd ~/Desktop/lerobot
pip install -e .
pip install pyrealsense2   # for the RealSense wrist/side cam (cam2)
```

Then apply this repo's patch on top, per [`so_arm_training/README.md`](../so_arm_training/README.md):

```bash
# from mpail-lerobot/
cp -r so_arm_training/async_inference/* ~/Desktop/lerobot/src/lerobot/async_inference/
```

## 2. Hardware setup & calibration (once per arm, in the `lerobot` env)

```bash
conda activate lerobot

# find the serial ports for the leader (teleop) and follower (physical) arms
lerobot-find-port

# calibrate the follower arm — writes to ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot_id>.json
lerobot-calibrate --robot.type=so_follower --robot.port=/dev/ttyACM0 --robot.id=<robot_id>

# if you're teleoperating with a leader arm to collect demos, calibrate it too
lerobot-calibrate --teleop.type=so_leader --teleop.port=/dev/ttyACM1 --teleop.id=<leader_id>
```

`<robot_id>` must match `--robot_id` used everywhere below (this repo's existing
calibration is `Kid`, leader is `Mom` — check
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/` for what's already calibrated).

### Sanity-check cameras and joints before running anything else

```bash
conda activate lerobot
lerobot-find-cameras   # list available opencv/realsense cameras
```

(This repo previously had `preview_cameras.py` / `read_joint_state.py` /
`wrist_snapshot.py` helper scripts for a quick visual/joint-state check before
running anything else — they're currently missing from the repo; recreate
them if you need this step.)

## 3. Start the robot-side gRPC server (`lerobot` env, keep running)

```bash
conda activate lerobot
python -m so_arm_training.so101_robot_server \
    --robot_port /dev/ttyACM0 --robot_id <robot_id> \
    --cam_index /dev/video0 --cam2_serial <realsense_serial> \
    --grpc_port 7070
```

Leave this running in its own terminal — it owns the arm and cameras for the
rest of the session. See the flags at the top of `so101_robot_server.py` for
servo tuning (`--p_coefficient`, `--i_coefficient`, `--goal_velocity`, etc.) if
motion is shaky or not settling.

## 4. Collect demonstrations, 5. Convert, 6. Train, 7. Evaluate

See [`so_arm_training/README.md`](../so_arm_training/README.md) for the full
walkthrough — demo collection (including the exact `robot_client` teleop
command and what each flag does), converting to `demo.pt`, training, and
evaluating a checkpoint.

## Where to read the algorithm code

Once the pipeline runs, see the main [README.md](../README.md)'s "About the
Files" section for where each MPAIL2 component (`encoder.py`, `dynamics.py`,
`reward.py`, `value.py`, `sampling.py`, `learner.py`) lives — `learner.py` is
the recommended starting point for reading the training logic.
