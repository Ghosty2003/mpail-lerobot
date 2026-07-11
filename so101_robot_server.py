"""
so101_robot_server.py — gRPC control server exposing the real SO-101 arm + cameras
to mpail2/envs/real/so101/so101_env.py.

Runs directly against lerobot's SOFollower driver — no LeRobot async_inference
layer involved, no separate client process required on this side. Mirrors
mpail2/envs/real/franka/network/server.py's role (thin RPC wrapper around the
hardware driver), but gRPC instead of ZMQ, and combines action+observation into
one round-trip per step instead of separate reads.

Run in the `lerobot` conda env, from the repo root (so `transport` is importable):

    conda activate lerobot
    python so101_robot_server.py \
        --robot_port /dev/ttyACM0 \
        --robot_id Kid \
        --cam_index /dev/video0 \
        --cam2_serial 317422074482 \
        --grpc_port 7070

Then point the mpail2-side env at it (mock=False):

    from mpail2.envs.real.so101 import SO101RealEnvArgs, make_so101_env
    env = make_so101_env(SO101RealEnvArgs(host="<this machine's IP>", port=7070, mock=False))
"""

import argparse
import logging
from concurrent import futures

import grpc
import numpy as np

from transport import so101_robot_pb2, so101_robot_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("so101_robot_server")

# Must match mpail2/envs/real/so101/robot_limits.py — JOINT_NAMES, HOME_POSITION_DEG,
# CAM_KEY/CAM2_KEY, CAM_H/CAM_W/CAM2_H/CAM2_W. Kept as local constants here since this
# script runs in the separate `lerobot` conda env, which doesn't have mpail2 installed.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME_POSITION_DEG = [8.0879, -6.0220, 2.9011, 87.7363, -0.2198, 30.0]
CAM_KEY, CAM2_KEY = "cam", "cam2"
CAM_H = CAM_W = 84
CAM2_H = CAM2_W = 84


def _joints_dict_to_array(obs: dict) -> np.ndarray:
    return np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)


def _to_bytes(img: np.ndarray, h: int, w: int) -> bytes:
    """uint8 (H0, W0, 3) camera frame -> raw bytes at (h, w, 3), resizing if needed."""
    img = np.asarray(img)
    if img.shape[0] != h or img.shape[1] != w:
        import cv2
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.uint8).tobytes()


class SO101RobotServicer(so101_robot_pb2_grpc.SO101RobotServicer):
    def __init__(self, robot):
        self.robot = robot

    def _state_from_obs(self, obs: dict) -> "so101_robot_pb2.RobotState":
        joints = _joints_dict_to_array(obs)
        cam = _to_bytes(obs[CAM_KEY], CAM_H, CAM_W) if CAM_KEY in obs else b""
        cam2 = _to_bytes(obs[CAM2_KEY], CAM2_H, CAM2_W) if CAM2_KEY in obs else b""
        return so101_robot_pb2.RobotState(joints_deg=joints.tolist(), cam=cam, cam2=cam2)

    def Reset(self, request, context):
        action = dict(zip([f"{n}.pos" for n in JOINT_NAMES], HOME_POSITION_DEG))
        self.robot.send_action(action)
        obs = self.robot.get_observation()
        logger.info("Reset -> home position")
        return self._state_from_obs(obs)

    def Step(self, request, context):
        joints = list(request.joints_deg)
        if len(joints) != len(JOINT_NAMES):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Expected {len(JOINT_NAMES)} joint targets, got {len(joints)}",
            )
        action = dict(zip([f"{n}.pos" for n in JOINT_NAMES], joints))
        self.robot.send_action(action)
        obs = self.robot.get_observation()
        return self._state_from_obs(obs)


def serve():
    parser = argparse.ArgumentParser(description="gRPC control server for the real SO-101 arm")
    parser.add_argument("--robot_port", default="/dev/ttyACM0", help="Serial port the arm is connected to")
    parser.add_argument("--robot_id", default="Kid", help="Calibration id (matches the LeRobot calibration file)")
    parser.add_argument("--cam_index", default="/dev/video0", help="Wrist cam (opencv) index or device path")
    parser.add_argument("--cam2_serial", default=None, help="RealSense serial number for cam2 (omit to disable)")
    parser.add_argument("--grpc_host", default="[::]")
    parser.add_argument("--grpc_port", type=int, default=7070)
    parser.add_argument(
        "--max_relative_target", type=float, default=None,
        help="Per-step delta clamp passed straight to lerobot's SOFollowerRobotConfig (default: no clamp).",
    )
    args = parser.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

    cameras = {CAM_KEY: OpenCVCameraConfig(index_or_path=args.cam_index, width=640, height=480, fps=30)}
    if args.cam2_serial:
        from lerobot.cameras.realsense import RealSenseCameraConfig
        cameras[CAM2_KEY] = RealSenseCameraConfig(
            serial_number_or_name=args.cam2_serial, width=640, height=480, fps=30
        )

    config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
    )
    robot = SO101Follower(config)
    logger.info(f"Connecting to SO-101 on {args.robot_port} (id={args.robot_id})...")
    robot.connect()
    logger.info("Connected.")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    so101_robot_pb2_grpc.add_SO101RobotServicer_to_server(SO101RobotServicer(robot), server)
    server.add_insecure_port(f"{args.grpc_host}:{args.grpc_port}")
    server.start()
    logger.info(f"so101_robot_server listening on {args.grpc_host}:{args.grpc_port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        robot.disconnect()
        server.stop(grace=2.0)


if __name__ == "__main__":
    serve()
