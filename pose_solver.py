import cv2
import numpy as np


# ==================== 目标实际尺寸 ====================

LED_LENGTH = 322.0      # 单条灯带长度，mm
LED_WIDTH = 3.4         # 单条灯带宽度，mm
LED_DISTANCE = 170.0    # 两条灯带中心间距，mm


# ==================== 目标三维模型 ====================

# 每条灯带：
# TL → TR → BR → BL
#
# 目标坐标系：
# X：左右
# Y：上下
# Z：垂直于灯板

OBJECT_POINTS = np.array(
    [
        # 左灯带
        [-(LED_DISTANCE + LED_WIDTH) / 2,  LED_LENGTH / 2, 0.0],  # TL
        [-(LED_DISTANCE - LED_WIDTH) / 2,  LED_LENGTH / 2, 0.0],  # TR
        [-(LED_DISTANCE - LED_WIDTH) / 2, -LED_LENGTH / 2, 0.0],  # BR
        [-(LED_DISTANCE + LED_WIDTH) / 2, -LED_LENGTH / 2, 0.0],  # BL

        # 右灯带
        [(LED_DISTANCE - LED_WIDTH) / 2,   LED_LENGTH / 2, 0.0],  # TL
        [(LED_DISTANCE + LED_WIDTH) / 2,   LED_LENGTH / 2, 0.0],  # TR
        [(LED_DISTANCE + LED_WIDTH) / 2,  -LED_LENGTH / 2, 0.0],  # BR
        [(LED_DISTANCE - LED_WIDTH) / 2,  -LED_LENGTH / 2, 0.0],  # BL
    ],
    dtype=np.float32
)


# ==================== PnP 解算 ====================

def solve_pose(image_points, K, dist):
    """
    根据目标的 2D 图像坐标和 3D 实际坐标，
    使用 solvePnP 计算目标相对于相机的位姿。

    参数：
        image_points:
            8 个图像坐标，顺序必须为：
            左灯带 TL, TR, BR, BL
            右灯带 TL, TR, BR, BL

        K:
            相机内参矩阵

        dist:
            相机畸变参数

    返回：
        success:
            是否解算成功

        rvec:
            目标相对于相机的旋转向量

        tvec:
            目标原点在相机坐标系中的位置，单位 mm
    """

    image_points = np.ascontiguousarray(
        image_points,
        dtype=np.float32
    ).reshape(8, 2)

    success, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    return success, rvec, tvec


# ==================== 位置提取 ====================

def get_position(tvec):
    """
    从 tvec 中提取 X、Y、Z。

    返回：
        position:
            [X, Y, Z]，单位 mm
    """

    return np.asarray(tvec, dtype=np.float64).reshape(3)


def position_to_meter(position):
    """
    mm → m
    """

    return np.asarray(position, dtype=np.float64) / 1000.0