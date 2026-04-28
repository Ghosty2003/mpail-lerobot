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
    1.9,         # gripper       ≈  22.2
], dtype=np.float32)

JOINT_UPPER_DEG = np.array([
    101.54,   # shoulder_pan  ≈ 150.9
    -10.81,   # shoulder_lift ≈ ...
    95.91,         # elbow_flex    ≈ 149.1
    74.54,         # wrist_flex    ≈ 110.6
    169.0,         # wrist_roll    ≈ 183.8 → cap at 180
    72.0,         # gripper       ≈ 156.6
], dtype=np.float32)

# Home position (degrees) — calibrated from physical robot.
# Order matches JOINT_NAMES: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
HOME_POSITION_DEG = np.array([16.44, -104.44, 95.91, 72.75, -0.04, 1.96], dtype=np.float32)

# ─── timing ───────────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS     = 200
CONTROL_FREQUENCY_HZ  = 1.0   # env step rate

# Maximum joint movement per control step (degrees).
# At 10 Hz: 5 deg/step = 50 deg/s.  Increase carefully — too high causes jerky motion.
MAX_DELTA_DEG = 10

# ─── LeRobot robot-control HTTP server ────────────────────────────────────────
# The env talks to a lightweight HTTP server running on the LeRobot side.
# Implement the server (see so101_env.py docstring for the expected API).
DEFAULT_ROBOT_HOST = "127.0.0.1"
DEFAULT_ROBOT_PORT = 7070       # intentionally different from ExternalPlannerServer (8080)
