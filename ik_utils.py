"""FK / IK utilities for the SO-101 arm using the SOA URDF and ikpy.

Chain layout (7 links):
  0  Base link   — fixed, inactive
  1  shoulder_pan
  2  shoulder_lift
  3  elbow_flex
  4  wrist_flex
  5  wrist_roll
  6  gripper      — not part of FK chain end-effector, inactive

IK method: Jacobian pseudo-inverse (differential IK), same principle as Franka's
internal Cartesian controller.  Iterates: Δq = J⁺ Δx until convergence or max_iter.
Uses damped least-squares (DLS) near singularities for numerical stability.

wrist_roll is excluded from the Jacobian — it does not affect EE position and is
controlled directly from the action (index 3 of the 5-dim action).
"""

import numpy as np

URDF_PATH = (
    "/home/chenmg93@netid.washington.edu"
    "/TECHIN517/ros2_ws/src/soa_ros2/soa_description/urdf/soa.urdf"
)

# Full chain mask — all joints active for FK (wrist_roll included so FK is correct).
_ACTIVE_MASK_FK = [False, True, True, True, True, True, False]

# Position-controlling joints only (indices into the 5-elem arm vector).
# wrist_roll (index 4) is excluded — it does not move the EE tip.
_POS_JOINTS = [0, 1, 2, 3]   # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        import ikpy.chain
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH, active_links_mask=_ACTIVE_MASK_FK)
    return _chain


def fk(arm_joints_deg: np.ndarray) -> np.ndarray:
    """Forward kinematics: 5 arm joint angles (degrees) → EE xyz (metres).

    Args:
        arm_joints_deg: shape (5,) — shoulder_pan … wrist_roll in degrees.

    Returns:
        xyz: shape (3,) float32 in metres.
    """
    chain = _get_chain()
    rad = np.deg2rad(arm_joints_deg.astype(np.float64))
    full = [0.0] + list(rad) + [0.0]
    T = chain.forward_kinematics(full)
    return T[:3, 3].astype(np.float32)


def fk_pose(arm_joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics: 5 arm joint angles (degrees) → EE xyz + quaternion.

    Args:
        arm_joints_deg: shape (5,) — shoulder_pan … wrist_roll in degrees.

    Returns:
        xyz:  shape (3,) float32 in metres.
        quat: shape (4,) float32 — [x, y, z, w] scalar-last convention.
    """
    from scipy.spatial.transform import Rotation as R
    chain = _get_chain()
    rad = np.deg2rad(arm_joints_deg.astype(np.float64))
    full = [0.0] + list(rad) + [0.0]
    T = chain.forward_kinematics(full)
    xyz = T[:3, 3].astype(np.float32)
    quat = R.from_matrix(T[:3, :3]).as_quat(scalar_first=False).astype(np.float32)  # [x,y,z,w]
    return xyz, quat


def _jacobian(arm_joints_deg: np.ndarray, eps_deg: float = 0.5) -> np.ndarray:
    """Numerical position Jacobian d(xyz)/d(q) for the 4 position-controlling joints.

    Returns shape (3, 4): columns are shoulder_pan, shoulder_lift, elbow_flex, wrist_flex.
    Units: metres per radian.
    """
    xyz0 = fk(arm_joints_deg).astype(np.float64)
    eps_rad = np.deg2rad(eps_deg)
    J = np.zeros((3, 4), dtype=np.float64)
    for col, joint_idx in enumerate(_POS_JOINTS):
        q_p = arm_joints_deg.copy().astype(np.float64)
        q_p[joint_idx] += eps_deg
        J[:, col] = (fk(q_p).astype(np.float64) - xyz0) / eps_rad
    return J


def ik(
    xyz_target: np.ndarray,
    initial_arm_deg: np.ndarray | None = None,
    max_iter: int = 8,
    tol_m: float = 0.002,
    damping: float = 0.05,
) -> np.ndarray:
    """Differential IK: Jacobian pseudo-inverse, same principle as Franka's controller.

    Iteratively steps joints along the Jacobian direction until the EE reaches
    xyz_target within tol_m, or max_iter is exhausted.  Uses damped least-squares
    (DLS) for stability near singularities.

    wrist_roll (index 4) is never modified — the caller sets it from the action.

    Args:
        xyz_target:      shape (3,) desired EE position in metres.
        initial_arm_deg: shape (5,) current joint angles in degrees (warm-start).
        max_iter:        maximum Jacobian iterations (default 8).
        tol_m:           convergence threshold in metres (default 2 mm).
        damping:         DLS damping factor λ — higher = more stable, less accurate.

    Returns:
        arm_joints_deg: shape (5,) float32 — shoulder_pan … wrist_roll.
    """
    if initial_arm_deg is None:
        initial_arm_deg = np.array([8.0879, -6.0220, 2.9011, 87.7363, -0.2198], np.float32)

    q = initial_arm_deg.copy().astype(np.float64)
    target = xyz_target.astype(np.float64)

    for _ in range(max_iter):
        error = target - fk(q).astype(np.float64)
        if np.linalg.norm(error) < tol_m:
            break

        J = _jacobian(q)                          # (3, 4)
        # Damped least-squares: J^T (J J^T + λ²I)^{-1}
        JJT = J @ J.T                             # (3, 3)
        J_dls = J.T @ np.linalg.inv(JJT + damping ** 2 * np.eye(3))  # (4, 3)

        dq_rad = J_dls @ error                    # (4,) in radians
        dq_deg = np.rad2deg(dq_rad)

        # Clamp per-step joint movement to avoid large jumps
        dq_deg = np.clip(dq_deg, -8.0, 8.0)

        for col, joint_idx in enumerate(_POS_JOINTS):
            q[joint_idx] += dq_deg[col]

    return q.astype(np.float32)
