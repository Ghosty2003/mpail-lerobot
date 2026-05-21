"""Physical constants and defaults for the SO-101 (SO-ARM101) 6-DOF robot arm."""

import numpy as np

# ─── dimensions ───────────────────────────────────────────────────────────────
STATE_DIM  = 6   # 6 joint positions reported by LeRobot
ACTION_DIM = 6   # 6 joint position targets sent to LeRobot

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
    -104.44,   # shoulder_lift
    -94.95,         # elbow_flex  
    -66.77,         # wrist_flex    ≈ -97.8
    -161.0,         # wrist_roll    ≈ -175.9
    10.0,        # gripper
], dtype=np.float32)

JOINT_UPPER_DEG = np.array([
    11.78,    # shoulder_pan — capped at home position (rightmost allowed)
    -10.81,   # shoulder_lift ≈ ...
    95.91,         # elbow_flex    ≈ 149.1
    74.54,         # wrist_flex    ≈ 110.6
    169.0,         # wrist_roll    ≈ 183.8 → cap at 180
    120.0,        # gripper
], dtype=np.float32)

# Home position (degrees) — calibrated from physical robot.
# Order matches JOINT_NAMES: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_POSITION_DEG = np.array([11.7802, -3.9121, 8.2637, 88.4396, 3.8242, 40.0], dtype=np.float32)

# ─── timing ───────────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS     = 99   # 100 client steps → 99 transitions, fills storage exactly
CONTROL_FREQUENCY_HZ  = 1.0   # env step rate

# Maximum joint movement per control step (degrees).
# At 10 Hz: 5 deg/step = 50 deg/s.  Increase carefully — too high causes jerky motion.
# Per-joint max movement per control step (degrees).
# Gripper gets a larger budget so it can open/close fully within a few steps.
MAX_DELTA_DEG = np.array([10, 10, 10, 10, 10, 30], dtype=np.float32)

# ─── LeRobot robot-control HTTP server ────────────────────────────────────────
# The env talks to a lightweight HTTP server running on the LeRobot side.
# Implement the server (see so101_env.py docstring for the expected API).
DEFAULT_ROBOT_HOST = "127.0.0.1"
DEFAULT_ROBOT_PORT = 7070       # intentionally different from ExternalPlannerServer (8080)

# ─── Camera ───────────────────────────────────────────────────────────────────
# The robot HTTP server must expose GET /camera → {"image": [[[r,g,b],...],...]}.
# The server should resize to (CAM_H, CAM_W) before sending.
CAM_KEY = "cam"   # must match the camera encoder_cfg obs_key
CAM_H   = 84
CAM_W   = 84
CAM_C   = 3
