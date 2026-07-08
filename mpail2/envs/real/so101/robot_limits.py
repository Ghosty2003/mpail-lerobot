"""Physical constants and defaults for the SO-101 (SO-ARM101) 6-DOF robot arm.

Action space: Cartesian end-effector (x, y, z) + gripper — 4-dim.
State space:  6 joint positions in degrees (unchanged).
IK converts the 4-dim policy action back to 6 joint targets before sending to the robot.
"""

import numpy as np

# ─── dimensions ───────────────────────────────────────────────────────────────
STATE_DIM      = 6   # 6 joint positions reported by LeRobot (raw, used for IK only)
ACTION_DIM     = 5   # Cartesian EE: [x, y, z, wrist_roll, gripper]
EE_PROPRIO_DIM = 13  # proprioception: [ee_x, ee_y, ee_z, qx, qy, qz, qw, j0..j5]

# Joint names in the order LeRobot reports / planner_server.py parses them
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint position bounds (degrees).  Tune to your calibration.
JOINT_LOWER_DEG = np.array([
    -51.16,   # shoulder_pan
    -26.02,   # shoulder_lift — home(-6.022) - 20
    -90.0,    # elbow_flex   — demo reaches -82.9°; extended from -17.1
    -66.77,   # wrist_flex
    -60.0,    # wrist_roll
    9.0,      # gripper — demo reaches 9.66°; lowered from 10.0
], dtype=np.float32)

JOINT_UPPER_DEG = np.array([
    35.0,     # shoulder_pan
    65.0,     # shoulder_lift — demo reaches 62.2°; extended from 13.98
    27.0,     # elbow_flex   — demo reaches 26.02°; raised from 22.90
    104.0,    # wrist_flex   — demo reaches 103.30°; raised from 102.5
    60.0,     # wrist_roll
    120.0,    # gripper
], dtype=np.float32)

# Home position (degrees) — calibrated from physical robot.
# Order matches JOINT_NAMES: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_POSITION_DEG = np.array([8.0879, -6.0220, 2.9011, 87.7363, -0.2198, 30.0], dtype=np.float32)

# ─── Cartesian workspace ──────────────────────────────────────────────────────
# Home EE position computed from FK at HOME_POSITION_DEG (arm joints only).
HOME_EE_M = np.array([0.2491, -0.0122, 0.1619], dtype=np.float32)  # metres

# Action space is DELTA (like Franka real): action_norm ∈ [-1,1]^3 is a per-step
# displacement scaled by MAX_DELTA_M * speed_scale.
# action=0 → hold current position (no home bias in MPPI roll-forward).
MAX_DELTA_M = np.float32(0.03)   # metres per step at speed_scale=1.0

# Hard workspace bounds — EE is clipped to this box after applying the delta.
EE_LOWER_M = np.array([0.14,  -0.075,  -0.01], dtype=np.float32)  # x: demo.pt reaches 0.1428m (1st pct 0.1496m); y: demo reaches -0.0716m (1st pct -0.0707m); z: demo reaches -0.0104m (1st pct -0.0099m)
EE_UPPER_M = np.array([0.45,   0.08,  0.21], dtype=np.float32)   # y: demo max +0.0776m (99th pct +0.0748m) + margin; x: covers 0.350m reach

# Kept for reference (not used in action mapping anymore).
EE_HALF_RANGE_M = (EE_UPPER_M - EE_LOWER_M) / 2

# Per-joint speed limit in degrees (legacy, used by so101_env.py joint-space path).
MAX_DELTA_DEG = np.array([10, 10, 10, 10, 10, 30], dtype=np.float32)

# Wrist roll action component (index 3): direct degree control.
HOME_WRIST_ROLL_DEG   = np.float32(-0.2198)   # matches HOME_POSITION_DEG[4]
WRIST_ROLL_HALF_RANGE = np.float32(5.0)        # max °/step at speed_scale=1.0 (delta action)

# Gripper action component (index 4): direct degree control.
GRIPPER_LOWER_DEG  = np.float32(10.0)
GRIPPER_UPPER_DEG  = np.float32(120.0)
HOME_GRIPPER_DEG   = np.float32(30.0)
GRIPPER_HALF_RANGE = np.float32(30.0)   # max °/step at speed_scale=1.0 (delta action) — demo closes ~44° in 20 steps

# ─── timing ───────────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS     = 299  # 300 client steps → 299 transitions, fills storage exactly
CONTROL_FREQUENCY_HZ  = 1.0   # env step rate

# ─── LeRobot robot-control HTTP server ────────────────────────────────────────
# The env talks to a lightweight HTTP server running on the LeRobot side.
# Implement the server (see so101_env.py docstring for the expected API).
DEFAULT_ROBOT_HOST = "127.0.0.1"
DEFAULT_ROBOT_PORT = 7070       # intentionally different from ExternalPlannerServer (8080)

# ─── Cameras ──────────────────────────────────────────────────────────────────
# CAM_KEY  = Web camera  (opencv, index_or_path=1, /dev/video1)
# CAM2_KEY = RealSense D435i  (serial 327743060231)
CAM_KEY  = "cam"    # Wrist camera — opencv /dev/video6
CAM_H    = 84
CAM_W    = 84
CAM_C    = 3

CAM2_KEY = "cam2"   # RealSense D435i — /dev/video0, serial 317422074482
CAM2_H   = 84
CAM2_W   = 84
CAM2_C   = 3
