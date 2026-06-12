import math
import numpy as np

np.set_printoptions(precision=4)

JOINTS = ["L1", "L2", "L3", "L4", "L5", "L6"]

JOINT_CONFIG = {
    "L1": {"home": 0,    "marlin": {"min": -97.4,  "max": 97.4},   "axis_sign":  1},
    "L2": {"home": -90,  "marlin": {"min": -140,   "max": -40},    "axis_sign":  1},
    "L3": {"home": 180,  "marlin": {"min": 110,    "max": 260},    "axis_sign": -1},
    "L4": {"home": 0,    "marlin": {"min": -142,   "max": 90},     "axis_sign": -1},
    "L5": {"home": 0,    "marlin": {"min": -128.2, "max": 120},    "axis_sign": -1},
    "L6": {"home": 180,  "marlin": {"min": 30,     "max": 300},    "axis_sign": -1},
}

HOME_ANGLES = {j: JOINT_CONFIG[j]["home"] for j in JOINTS}

DEFAULT_CONVENTION = "urdf"          
CONVENTIONS = ("urdf", "diagram")

TOOL_FIX = np.array([[1.0,  0.0,  0.0, 0.0],
                     [0.0, -1.0,  0.0, 0.0],
                     [0.0,  0.0, -1.0, 0.0],
                     [0.0,  0.0,  0.0, 1.0]], dtype=float)


def _resolve_convention(convention):
    c = DEFAULT_CONVENTION if convention is None else convention
    if c not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {c!r}")
    return c

def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]], dtype=float)

def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], dtype=float)

def _Tr(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T

def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF RPY convention: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


def rpy_from_matrix(R: np.ndarray) -> tuple:
    """
    Analytic inverse of _rpy(): recover (roll, pitch, yaw) in radians from a
    3x3 (or 4x4) rotation matrix built as  R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Derivation (ZYX). For that product the matrix has the form
        R[2,0] = -sin(pitch)
        R[2,1] =  cos(pitch) sin(roll)
        R[2,2] =  cos(pitch) cos(roll)
        R[0,0] =  cos(yaw)   cos(pitch)
        R[1,0] =  sin(yaw)   cos(pitch)
    so, away from the singularity,
        pitch = atan2(-R[2,0], hypot(R[0,0], R[1,0]))
        roll  = atan2( R[2,1], R[2,2])
        yaw   = atan2( R[1,0], R[0,0])

    Gimbal lock: when pitch = +-90deg, cos(pitch) = 0 and the entries that
    roll/yaw rely on all vanish, so only (roll -+ yaw) is defined. We pin
    yaw = 0 and fold the free rotation into roll (the same choice the SciPy
    decomposition makes, up to an equivalent 180deg split).
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    r20 = float(R[2, 0])
    eps = 1e-9

    if r20 <= -1.0 + eps:                       # pitch = +90 deg
        pitch = math.pi / 2.0
        yaw = 0.0
        roll = math.atan2(R[0, 1], R[0, 2])
    elif r20 >= 1.0 - eps:                       # pitch = -90 deg
        pitch = -math.pi / 2.0
        yaw = 0.0
        roll = math.atan2(-R[0, 1], -R[0, 2])
    else:                                        # regular case
        pitch = math.atan2(-r20, math.hypot(R[0, 0], R[1, 0]))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


def rpy_from_matrix_deg(R: np.ndarray) -> dict:
    """(roll, pitch, yaw) in degrees as a labelled dict."""
    roll, pitch, yaw = rpy_from_matrix(R)
    return {
        "roll":  math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw":   math.degrees(yaw),
    }


def _joint_transform(
    ox: float, oy: float, oz: float,
    roll: float, pitch: float, yaw: float,
    axis_sign: int,
    q_rad: float,
) -> np.ndarray:
    return _Tr(ox, oy, oz) @ _rpy(roll, pitch, yaw) @ _Rz(axis_sign * q_rad)

def marlin_to_urdf_rad(joint: str, marlin_deg: float) -> float:
    cfg = JOINT_CONFIG[joint]
    return cfg["axis_sign"] * (marlin_deg - cfg["home"]) * (math.pi / 180.0)

def fk_all_frames(marlin_angles: dict, convention: str = None) -> list[np.ndarray]:
    convention = _resolve_convention(convention)
    q = {j: marlin_to_urdf_rad(j, marlin_angles[j]) for j in JOINTS}

    DEFS = [
        (0.0,        0.0,    0.0,       0.0,          0.0,   0.0,           1),   # L1
        (0.02342072, 0.0,    0.1105,   -math.pi/2,    0.0,   0.0,           1),   # L2
        (0.0,       -0.18,   0.0,       math.pi,      0.0,  -math.pi/2,    -1),   # L3
        (0.0435,     0.0,    0.0,       math.pi/2,    0.0,   math.pi,      -1),   # L4
        (0.0,        0.0,   -0.17635,  -math.pi/2,    0.0,   0.0,          -1),   # L5
        (0.0,        0.0,    0.0,       math.pi/2,    0.0,   0.0,          -1),   # L6
    ]

    frames = [np.eye(4)]  # T_world_base
    T = np.eye(4)
    for (tx, ty, tz, r, p, y, sgn), joint in zip(DEFS, JOINTS):
        T = T @ _joint_transform(tx, ty, tz, r, p, y, sgn, q[joint])
        frames.append(T.copy())

    if convention == "diagram":
        frames[-1] = frames[-1] @ TOOL_FIX   

    return frames


def fk(marlin_angles: dict, convention: str = None) -> np.ndarray:
    return fk_all_frames(marlin_angles, convention)[-1]

def _compute_R_home(convention: str = None) -> np.ndarray:
    return fk(HOME_ANGLES, convention)[:3, :3].copy()

R_HOME = {c: _compute_R_home(c) for c in CONVENTIONS}

def fk_pose(marlin_angles: dict, convention: str = None) -> dict:
    
    from scipy.spatial.transform import Rotation as _Rot
    convention = _resolve_convention(convention)
    T = fk(marlin_angles, convention)
    p = T[:3, 3] * 1000.0
    R_abs = T[:3, :3]
    R_rel = R_HOME[convention].T @ R_abs
    r = _Rot.from_matrix(R_rel)
    euler = r.as_euler('ZYX', degrees=True)   
    quat  = r.as_quat()                        

    roll_a, pitch_a, yaw_a = rpy_from_matrix(R_abs)

    return {
        "position_mm":   np.round(p, 4).tolist(),
        "R_abs":         np.round(R_abs, 4),
        "R_rel":         np.round(R_rel, 4),
        "euler_rel_deg": np.round(euler, 4).tolist(),     # [yaw, pitch, roll], relative to home
        "euler_abs_deg": [round(math.degrees(yaw_a), 4),  # [yaw, pitch, roll], absolute (world)
                          round(math.degrees(pitch_a), 4),
                          round(math.degrees(roll_a), 4)],
        "quaternion":    np.round(quat, 4).tolist(),
        "tool_axis":     np.round(R_abs[:, 2], 4).tolist(),
        "convention":    convention,
    }