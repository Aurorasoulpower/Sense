import os
import sys

# 让 test/ 下的工具能 import 根目录的模块（camera 等）
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import cv2
import numpy as np

from camera import Camera


# ==================================================
# 方案 B：白色过曝灯芯检测（默认阈值）
# ==================================================

MIN_VAL_DEFAULT = 220   # R/G/B 三个通道都要 >= 这个值（够亮）
MAX_DIFF_DEFAULT = 30   # R/G/B 之间最大差 <= 这个值（接近白色）


def _on_trackbar(_):
    pass


def _rgb_on_click(event, x, y, flags, param):
    """左键点击打印该点 BGR 值，方便定白色灯芯阈值。"""
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            b, g, r = frame[y, x]
            print(f"({x},{y}) B={b} G={g} R={r}")


def _white_core_mask(frame, min_val, max_diff):
    """
    高亮 + 接近白色 的二值掩膜。

    等价于：
        (R/G/B 都 >= min_val) 且 (R/G/B 两两接近)
    用 min/max 通道实现，比逐对做 abs 更简洁，且顺带覆盖了 g-b 这一对。
    """

    b, g, r = cv2.split(frame)

    min_ch = np.minimum(np.minimum(r, g), b)
    max_ch = np.maximum(np.maximum(r, g), b)

    bright = min_ch >= min_val                    # 三个通道都够亮
    near_white = (max_ch - min_ch) <= max_diff    # 三个通道接近（接近白）

    mask = (bright & near_white).astype(np.uint8) * 255

    # 形态学闭运算：连接可能断开的灯芯段
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main():

    camera = Camera()
    camera.print_settings()

    win = "raw"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("min_val(亮度下限)", win, MIN_VAL_DEFAULT, 255, _on_trackbar)
    cv2.createTrackbar("max_diff(接近白)", win, MAX_DIFF_DEFAULT, 255, _on_trackbar)

    mouse_param = {"frame": None}
    cv2.setMouseCallback(win, _rgb_on_click, mouse_param)

    print("滑杆调白色灯芯阈值，按 q / Esc 退出；鼠标左键点画面查看 BGR")

    while True:

        frame = camera.read()
        if frame is None:
            continue

        mouse_param["frame"] = frame

        min_val = cv2.getTrackbarPos("min_val(亮度下限)", win)
        max_diff = cv2.getTrackbarPos("max_diff(接近白)", win)

        white_mask = _white_core_mask(frame, min_val, max_diff)

        # 方案 A：红色掩膜，用于 A/B 对比
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 100, 100), (20, 255, 255))

        # 白色灯芯像素占比
        white_ratio = (white_mask > 0).mean() * 100.0

        display = frame.copy()
        lines = [
            f"min_val={min_val}  max_diff={max_diff}",
            f"white-core={white_ratio:.2f}%",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                display,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(win, display)
        cv2.imshow("white_core", white_mask)
        cv2.imshow("red (A/B)", red_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
