import cv2
import pyrealsense2 as rs
import numpy as np
import math

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

# 获取 RGB 相机的 profile
color_profile = profile.get_stream(
    rs.stream.color
).as_video_stream_profile()

# 获取相机内参
intrinsics = color_profile.get_intrinsics()

print("fx =", intrinsics.fx)
print("fy =", intrinsics.fy)
print("cx =", intrinsics.ppx)
print("cy =", intrinsics.ppy)

print("畸变模型 =", intrinsics.model)
print("畸变参数 =", intrinsics.coeffs)

pipeline.stop()
# ==================================================
# 1. 创建 RealSense 管线
# ==================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

pipeline.start(config)


# ==================================================
# 2. 灯带方向计算函数
# ==================================================

def get_long_side_angle(box):

    max_length = 0
    best_angle = 0

    for i in range(4):

        p1 = box[i]
        p2 = box[(i + 1) % 4]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = (dx ** 2 + dy ** 2) ** 0.5

        if length > max_length:

            max_length = length

            best_angle = math.degrees(
                math.atan2(dy, dx)
            )

    return best_angle


# ==================================================
# 3. 实时处理
# ==================================================

try:

    while True:

        # --------------------------------------------------
        # 3.1 获取 RealSense 当前帧
        # --------------------------------------------------

        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        img = np.asanyarray(
            color_frame.get_data()
        )


        # --------------------------------------------------
        # 3.2 BGR → HSV
        # --------------------------------------------------

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )


        # --------------------------------------------------
        # 3.3 颜色阈值
        # --------------------------------------------------

        lower = (0, 100, 100)
        upper = (20, 255, 255)

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )


        # --------------------------------------------------
        # 3.4 形态学处理
        # --------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 42)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # --------------------------------------------------
        # 3.5 寻找轮廓
        # --------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # --------------------------------------------------
        # 3.6 几何筛选
        # --------------------------------------------------

        candidates = []

        result = img.copy()

        for contour in contours:

            rect = cv2.minAreaRect(contour)

            (cx, cy), (w, h), angle = rect

            long_side = max(w, h)
            short_side = min(w, h)

            if short_side == 0:
                continue

            aspect_ratio = (
                long_side / short_side
            )

            # 长宽比筛选
            if aspect_ratio < 10:
                continue

            # 获取四个角点
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            candidate = {
                "contour": contour,
                "center": (cx, cy),
                "size": (w, h),
                "angle": angle,
                "aspect_ratio": aspect_ratio,
                "box": box
            }

            candidates.append(candidate)


        # --------------------------------------------------
        # 3.7 绘制所有候选灯带
        # --------------------------------------------------

        for i, candidate in enumerate(candidates):

            box = candidate["box"]

            cx, cy = candidate["center"]

            # 绘制矩形
            cv2.drawContours(
                result,
                [box],
                0,
                (0, 0, 255),
                2
            )

            # 绘制中心
            cv2.circle(
                result,
                (int(cx), int(cy)),
                5,
                (255, 0, 0),
                -1
            )

            # 编号
            cv2.putText(
                result,
                f"LED {i}",
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # 角点
            for j, point in enumerate(box):

                px, py = point

                cv2.circle(
                    result,
                    (px, py),
                    4,
                    (0, 255, 255),
                    -1
                )


        # ==================================================
        # 4. 目标几何信息
        # ==================================================

        if len(candidates) >= 2:

            led1 = candidates[0]
            led2 = candidates[1]

            # --------------------------------------------------
            # 两条灯带中心
            # --------------------------------------------------

            cx1, cy1 = led1["center"]
            cx2, cy2 = led2["center"]

            # --------------------------------------------------
            # 目标中心
            # --------------------------------------------------

            target_cx = (
                cx1 + cx2
            ) / 2

            target_cy = (
                cy1 + cy2
            ) / 2

            # --------------------------------------------------
            # 两灯带中心距离
            # --------------------------------------------------

            dx = cx2 - cx1
            dy = cy2 - cy1

            center_distance = (
                dx ** 2 + dy ** 2
            ) ** 0.5

            # --------------------------------------------------
            # 灯带方向
            # --------------------------------------------------

            led1_angle = get_long_side_angle(
                led1["box"]
            )

            led2_angle = get_long_side_angle(
                led2["box"]
            )

            target_angle = (
                led1_angle + led2_angle
            ) / 2

            # --------------------------------------------------
            # 绘制目标中心
            # --------------------------------------------------

            cv2.circle(
                result,
                (
                    int(target_cx),
                    int(target_cy)
                ),
                8,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                result,
                "TARGET",
                (
                    int(target_cx) + 10,
                    int(target_cy)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # 绘制两灯带中心连线
            # --------------------------------------------------

            cv2.line(
                result,
                (
                    int(cx1),
                    int(cy1)
                ),
                (
                    int(cx2),
                    int(cy2)
                ),
                (255, 0, 255),
                2
            )

            # --------------------------------------------------
            # 实时显示数据
            # --------------------------------------------------

            cv2.putText(
                result,
                f"Center: ({target_cx:.1f}, {target_cy:.1f})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                result,
                f"Distance: {center_distance:.1f} px",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                result,
                f"Angle: {target_angle:.1f} deg",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        # ==================================================
        # 5. 显示
        # ==================================================

        cv2.imshow(
            "RealSense RGB",
            result
        )

        cv2.imshow(
            "Mask",
            mask
        )


        # ==================================================
        # 6. 按 q 退出
        # ==================================================

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:

    pipeline.stop()
    cv2.destroyAllWindows()