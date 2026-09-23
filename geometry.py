import math
import numpy as np
from dataclasses import dataclass

from detector import LightBar


# ==================================================
# 阈值（可调）
# ==================================================

LENGTH_DIFF_MAX = 0.15
WIDTH_DIFF_MAX = 0.30
ANGLE_DIFF_MAX = 10.0            # deg
PERPENDICULAR_ERR_MAX = 12.0     # deg
SIDE_RATIO_MAX = 0.20


# ==================================================
# 数据对象
# ==================================================

@dataclass
class Target:
    """经过几何验证的目标：左右两根灯带 + 4 个图像点。"""

    left: LightBar
    right: LightBar
    image_points: np.ndarray     # (4, 2) [L_T, R_T, R_B, L_B]


# ==================================================
# 角度工具
# ==================================================

def _normalize_angle(a):
    """角度归一化到 [0, 180)。"""
    return a % 180.0


def _angle_diff(a, b):
    """两条直线方向角的最小夹角（0~90°），处理 180° 周期性。"""
    d = abs(_normalize_angle(a) - _normalize_angle(b))
    return 180.0 - d if d > 90.0 else d


def get_long_side_angle(corners):
    """灯带长边方向角（deg），取四角点中最长边的角度。"""

    max_length = 0.0
    best_angle = 0.0

    for i in range(4):

        p1 = corners[i]
        p2 = corners[(i + 1) % 4]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = math.hypot(dx, dy)

        if length > max_length:
            max_length = length
            best_angle = math.degrees(
                math.atan2(dy, dx)
            )

    return best_angle


def _distance(p1, p2):
    """两点欧氏距离。"""
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


# ==================================================
# 目标匹配
# ==================================================

def match_light_bars(b1, b2):
    """
    校验两根灯带是否构成一个合法目标。

    返回：
        ok : bool
        reason : str
    """

    # ① 长边相似
    length_diff = abs(b1.height - b2.height) / max(b1.height, b2.height)

    if length_diff > LENGTH_DIFF_MAX:
        return False, f"length_diff={length_diff:.2f}"


    # ② 短边相似
    width_diff = abs(b1.width - b2.width) / max(b1.width, b2.width)

    if width_diff > WIDTH_DIFF_MAX:
        return False, f"width_diff={width_diff:.2f}"


    # ③ 方向相似（近似平行）
    dir1 = get_long_side_angle(b1.corners)
    dir2 = get_long_side_angle(b2.corners)
    angle_diff = _angle_diff(dir1, dir2)

    if angle_diff > ANGLE_DIFF_MAX:
        return False, f"angle_diff={angle_diff:.1f}"


    # ④ 中心连线 ⊥ 灯带方向
    cx1, cy1 = b1.center
    cx2, cy2 = b2.center

    center_line_angle = math.degrees(
        math.atan2(cy2 - cy1, cx2 - cx1)
    )

    perpendicular_error = abs(
        _angle_diff(center_line_angle, dir1) - 90.0
    )

    if perpendicular_error > PERPENDICULAR_ERR_MAX:
        return False, f"perp_error={perpendicular_error:.1f}"


    return True, "ok"


# ==================================================
# 四点提取
# ==================================================

def _strip_endpoints(corners):
    """
    取灯带的顶部中心、底部中心。

    四角点按 y 排序，y 小的两角中点为顶部中心，y 大的两角中点为底部中心。

    返回：
        top : (x, y)
        bottom : (x, y)
    """

    corners_sorted = sorted(
        corners,
        key=lambda p: p[1]
    )

    top_a, top_b = corners_sorted[0], corners_sorted[1]
    bottom_a, bottom_b = corners_sorted[2], corners_sorted[3]

    top = (
        (top_a[0] + top_b[0]) / 2.0,
        (top_a[1] + top_b[1]) / 2.0,
    )

    bottom = (
        (bottom_a[0] + bottom_b[0]) / 2.0,
        (bottom_a[1] + bottom_b[1]) / 2.0,
    )

    return top, bottom


def build_target_points(left, right):
    """
    由左右两根灯带提取 4 个稳定中心点，并做结构校验。

    返回：
        points : np.ndarray (4, 2)
            [L_T, R_T, R_B, L_B]，对应目标 TL/TR/BR/BL。
            校验失败返回 None。
    """

    L_T, L_B = _strip_endpoints(left.corners)
    R_T, R_B = _strip_endpoints(right.corners)


    # 结构校验：上边 ≈ 下边
    top = _distance(L_T, R_T)
    bottom = _distance(L_B, R_B)

    if abs(top - bottom) / top > SIDE_RATIO_MAX:
        return None


    # 结构校验：左边 ≈ 右边
    left_side = _distance(L_T, L_B)
    right_side = _distance(R_T, R_B)

    if abs(left_side - right_side) / left_side > SIDE_RATIO_MAX:
        return None


    points = np.array(
        [L_T, R_T, R_B, L_B],
        dtype=np.float32
    )

    return points


# ==================================================
# 目标生成
# ==================================================

def match_target(candidates):
    """
    从候选中挑出最像的两根，验证是否构成合法目标，并提取 4 个中心点。

    返回：
        target : Target | None
    """

    if len(candidates) < 2:
        return None

    b1, b2 = candidates[0], candidates[1]

    ok, _ = match_light_bars(b1, b2)

    if not ok:
        return None


    # 左右判断：center.x 小的在左
    if b1.center[0] < b2.center[0]:
        left, right = b1, b2
    else:
        left, right = b2, b1


    image_points = build_target_points(left, right)

    if image_points is None:
        return None

    return Target(
        left=left,
        right=right,
        image_points=image_points,
    )
