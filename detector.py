import cv2
import numpy as np
from dataclasses import dataclass


# ==================================================
# 阈值
# ==================================================

RED_LOWER = (0, 100, 100)
RED_UPPER = (20, 255, 255)

WHITE_MIN_VAL = 120   # 白芯：R/G/B 三通道最小值 >= 该值（灯带本体）

MIN_ASPECT_RATIO = 10.0


# ==================================================
# 数据对象
# ==================================================

@dataclass
class LightBar:
    """一根灯带的几何信息。"""

    center: tuple           # (cx, cy)
    width: float            # 短边，像素
    height: float           # 长边，像素
    angle: float            # minAreaRect 角度，deg
    corners: np.ndarray     # (4, 2) 四角点


# ==================================================
# 单灯带检测
# ==================================================

def detect_light_bars(img, mode="white"):
    """
    图像 → 掩膜 → 形态学闭运算 → 轮廓 → minAreaRect → 长宽比筛选。

    mode：
        "white" 白芯掩膜（默认）：min(R,G,B) >= WHITE_MIN_VAL，
                直接框出灯带本体，避免红色光晕过曝成白造成的黑洞。
        "red"   红色掩膜：用于 A/B 对比。

    返回：
        candidates : list[LightBar]
            按 height（长边）降序排列。

        mask : np.ndarray
            二值掩膜，供 main 显示。
    """

    # --------------------------------------------------
    # 1. 掩膜（白芯 / 红色）
    # --------------------------------------------------

    if mode == "red":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 42))
    else:
        min_ch = np.min(img, axis=2)
        mask = (min_ch >= WHITE_MIN_VAL).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))


    # --------------------------------------------------
    # 2. 形态学闭运算：桥接灯带沿长边的断点（白芯因 PWM 频闪略有断续）
    # --------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------
    # 3. 寻找轮廓
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------
    # 4. minAreaRect + 长宽比筛选
    # --------------------------------------------------

    candidates = []

    for contour in contours:

        rect = cv2.minAreaRect(contour)

        (cx, cy), (w, h), angle = rect

        width = min(w, h)
        height = max(w, h)

        if width == 0:
            continue

        if height / width < MIN_ASPECT_RATIO:
            continue

        corners = cv2.boxPoints(rect)

        candidates.append(
            LightBar(
                center=(cx, cy),
                width=width,
                height=height,
                angle=angle,
                corners=corners,
            )
        )


    # 最长的两根最像真灯带
    candidates.sort(
        key=lambda bar: bar.height,
        reverse=True
    )


    return candidates, mask
