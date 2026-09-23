import os
import sys

# 让 test/ 下的工具能 import 根目录的模块（camera 等）
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import cv2
import numpy as np
import pyrealsense2 as rs

from camera import Camera


# 与 detector.py 保持一致的红色阈值，调参时用来目测灯带被覆盖的情况
RED_LOWER = (0, 100, 100)
RED_UPPER = (20, 255, 255)


def _on_trackbar(_):
    """滑杆回调：什么都不做，主循环里每帧读取一次当前值即可。"""
    pass


def _hsv_on_click(event, x, y, flags, param):
    """鼠标左键点击画面，打印该点 HSV 值。"""
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            h, s, v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
            print(f"({x},{y}) H={h} S={s} V={v}")


def main():

    camera = Camera()
    camera.print_settings()

    if camera.sensor is None:
        print("未找到彩色传感器，无法调参")
        return

    # 读取各选项真实范围，用于设置滑杆
    exp_rng = camera.sensor.get_option_range(rs.option.exposure)
    gain_rng = camera.sensor.get_option_range(rs.option.gain)
    wb_rng = camera.sensor.get_option_range(rs.option.white_balance)

    # 曝光只需要低端：灯带很亮，上限取 300（约 30ms）已足够
    exp_max = int(min(exp_rng.max, 300))

    exp_init = int(min(camera.get_option(rs.option.exposure), exp_max))
    gain_init = int(camera.get_option(rs.option.gain))
    wb_init = int(camera.get_option(rs.option.white_balance))

    win = "raw"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("exposure(100us)", win, exp_init, exp_max, _on_trackbar)
    cv2.createTrackbar("gain", win, gain_init, int(gain_rng.max), _on_trackbar)
    cv2.createTrackbar("white_balance(K)", win, wb_init, int(wb_rng.max), _on_trackbar)

    mouse_param = {"frame": None}
    cv2.setMouseCallback(win, _hsv_on_click, mouse_param)

    print("滑杆调参，按 q / Esc 退出；鼠标左键点画面查看该点 HSV")

    while True:

        frame = camera.read()
        if frame is None:
            continue

        mouse_param["frame"] = frame

        # 每帧把滑杆值写回相机
        exp = cv2.getTrackbarPos("exposure(100us)", win)
        gain = cv2.getTrackbarPos("gain", win)
        wb = cv2.getTrackbarPos("white_balance(K)", win)

        camera.set_option(rs.option.exposure, exp)
        camera.set_option(rs.option.gain, gain)
        camera.set_option(rs.option.white_balance, wb)

        # 红色掩膜（与 detector.py 一致）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)

        # 近白像素比例：过曝的灯带中心会变成接近白色，这个值越高说明越爆
        near_white = np.all(frame >= 250, axis=2)
        blow_ratio = near_white.mean() * 100.0

        display = frame.copy()
        lines = [
            f"exposure={exp}  gain={gain}  wb={wb}",
            f"near-white={blow_ratio:.2f}%  (过曝越严重这个值越高)",
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
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
