import math
import numpy as np

JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6"]

JOINT_CONFIG = {
    "L1": {"home": 0,    "marlin": {"min": -170,  "max": 100},   "axis_sign":  1},
    "L2": {"home": -90,  "marlin": {"min": -140,   "max": -40},    "axis_sign":  1},
    "L3": {"home": 180,  "marlin": {"min": 110,    "max": 260},    "axis_sign": -1},
    "L4": {"home": 0,    "marlin": {"min": -142,   "max": 90},     "axis_sign": -1},
    "L5": {"home": 0,    "marlin": {"min": -128.2, "max": 120},    "axis_sign": -1},
    "L6": {"home": 180,  "marlin": {"min": 30,     "max": 300},    "axis_sign": -1},
}

HOME_ANGLES = {j: JOINT_CONFIG[j]["home"] for j in JOINTS}
DEFAULT_CONVENTION = "diagram"
CONVENTIONS = ("urdf", "diagram")

TOOL_FIX = np.array([[1.0,  0.0,  0.0, 0.0],
                     [0.0, -1.0,  0.0, 0.0],
                     [0.0,  0.0, -1.0, 0.0],
                     [0.0,  0.0,  0.0, 1.0]], dtype=float)

R_HOME = {c: _compute_R_home(c) for c in CONVENTIONS}

def _compute_R_home(convention: str = None) -> np.ndarray:
    return fk(HOME_ANGLES, convention)[:3, :3].copy()


def _resolve_convention(convention):
    c = DEFAULT_CONVENTION if convention is None else convention
    if c not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {c}")
    return c

def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0 ],
                     [0, c, -s, 0],
                     [0, s, c, 0 ],
                     [0, 0, 0, 1]], dtype=float)

def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)

def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)

def _Tr(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T

def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)

def _joint_transform(ox: float, oy: float, oz: float,
                    roll: float, pitch: float, yaw: float,
                    axis_sign: int, q_rad: float) -> np.ndarray:
    
    return _Tr(ox, oy, oz) @ _rpy(roll, pitch, yaw) @ _Rz(axis_sign * q_rad)

def marlin_to_urdf_rad(joint: str, marlin_deg: float) -> float:
    cfg = JOINT_CONFIG[joint]
    return cfg["axis_sign"] * (marlin_deg - cfg["home"]) * (math.pi / 180.0)

def fk_all_frames(marlin_angles: dict, convention: str = None) -> list[np.ndarray]:
    convention = _resolve_convention(convention)
    q = {j: marlin_to_urdf_rad(j, marlin_angles[j]) for j in JOINTS}
 
    #   (tx,         ty,     tz,        roll,        pitch, yaw,    axis_sign)
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

def fk_pose(marlin_angles: dict, convention: str = None) -> dict:

    from scipy.spatial.transform import Rotation as _Rot
    convention = _resolve_convention(convention)
    T = fk(marlin_angles, convention)
    p = T[:3, 3] * 1000.0
    R_abs = T[:3, :3]
    R_rel = R_HOME[convention],T @ R_abs
    r = _Rot.from_matrix(R_rel)
    euler = r.as_euler('ZYX', degrees=True)
    quat = r.as_quat()

    return {
        "position_mm":   p.tolist(),
        "R_abs":         R_abs,
        "R_rel":         R_rel,
        "euler_rel_deg": euler.tolist(),       
        "quaternion":    quat.tolist(),         
        "tool_axis":     R_abs[:, 2].tolist(),  
        "convention":    convention,

    }