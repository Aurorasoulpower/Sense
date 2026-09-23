import math
import cv2
import numpy as np
from dataclasses import dataclass


# ==================== 目标实际尺寸 ====================

LED_LENGTH = 322.0      # 单条灯带长度，mm
LED_DISTANCE = 170.0    # 两条灯带中心间距，mm


# ==================== 目标三维模型 ====================

# 4 个灯带中心线端点（顶部/底部中心），目标原点在两灯带中心
# 顺序：TL → TR → BR → BL
OBJECT_POINTS = np.array(
    [
        [-LED_DISTANCE / 2,  LED_LENGTH / 2, 0.0],   # TL 左顶中心
        [ LED_DISTANCE / 2,  LED_LENGTH / 2, 0.0],   # TR 右顶中心
        [ LED_DISTANCE / 2, -LED_LENGTH / 2, 0.0],   # BR 右底中心
        [-LED_DISTANCE / 2, -LED_LENGTH / 2, 0.0],   # BL 左底中心
    ],
    dtype=np.float32
)


REPROJ_ERROR_MAX = 5.0  # px


# ==================== 数据对象 ====================

@dataclass
class Pose:
    """目标相对相机的位姿。"""

    rvec: np.ndarray             # (3, 1)
    tvec: np.ndarray             # (3, 1) 单位 mm
    position: np.ndarray         # (3,)   [X, Y, Z] 单位 m
    roll: float                  # deg
    pitch: float                 # deg
    yaw: float                   # deg
    reprojection_error: float    # px
    valid: bool                  # 重投影误差是否在阈值内


# ==================== 工具 ====================

def _euler_angles(rvec):
    """从旋转向量提取 ZYX 欧拉角（外旋，deg）。"""

    R, _ = cv2.Rodrigues(rvec)

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = 0.0

    return roll, pitch, yaw


def _reprojection_error(image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """平均重投影像素误差。"""

    image_points = np.asarray(
        image_points,
        dtype=np.float32
    ).reshape(4, 2)

    projected, _ = cv2.projectPoints(
        OBJECT_POINTS,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    projected = projected.reshape(-1, 2)

    error = np.mean(
        np.linalg.norm(
            image_points - projected,
            axis=1
        )
    )

    return float(error)


# ==================== PnP 解算 ====================

def solve_pose(image_points, camera_matrix, dist_coeffs):
    """
    4 点共面 PnP（SOLVEPNP_IPPE）。

    返回：
        pose : Pose | None
            解算失败返回 None。
    """

    image_points = np.ascontiguousarray(
        image_points,
        dtype=np.float32
    ).reshape(4, 2)

    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        OBJECT_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if retval == 0 or len(rvecs) == 0:
        return None


    # IPPE 返回两个解（不同版本 solvePnP 可能只回传单个）；选相机前方（Z>0）
    # 的解，都不在则选重投影误差更小的。
    rvecs = [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs]
    tvecs = [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs]

    z = [t[2] for t in tvecs]

    if any(v > 0 for v in z):
        idx = int(np.argmax(z))
    else:
        errs = [
            _reprojection_error(
                image_points,
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            for i in range(len(rvecs))
        ]
        idx = int(np.argmin(errs))


    rvec = rvecs[idx].reshape(3, 1)
    tvec = tvecs[idx].reshape(3, 1)

    reproj = _reprojection_error(
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    position = np.asarray(
        tvec,
        dtype=np.float64
    ).reshape(3) / 1000.0   # mm → m

    roll, pitch, yaw = _euler_angles(rvec)

    return Pose(
        rvec=rvec,
        tvec=tvec,
        position=position,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        reprojection_error=reproj,
        valid=reproj <= REPROJ_ERROR_MAX,
    )
