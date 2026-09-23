import cv2
import numpy as np

from camera import Camera
from detector import detect_light_bars
from geometry import match_target
from pose import solve_pose


# ==================================================
# 可视化：每根灯带 → 旋转矩形框 + 4 个角点
# ==================================================

def draw_light_bars(result, candidates):
    """画出每根灯带：旋转矩形框、4 个角点、中心点。"""

    for i, bar in enumerate(candidates):

        corners = bar.corners.astype(int)
        cx, cy = bar.center

        # 旋转矩形框
        cv2.drawContours(result, [corners], 0, (0, 0, 255), 2)

        # 4 个角点
        for j, (px, py) in enumerate(corners):
            cv2.circle(result, (px, py), 4, (0, 255, 255), -1)
            cv2.putText(
                result,
                f"c{j}",
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # 中心点
        cv2.circle(result, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.putText(
            result,
            f"LED {i}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


# ==================================================
# 信息面板：当前目标位置 + 相对初始位置的位移
# ==================================================

def render_info_panel(pose, init_position):
    """生成信息面板图像，每帧刷新显示位置与位移。"""

    panel = np.zeros((320, 460, 3), dtype=np.uint8)

    def put(text, y, color=(255, 255, 255)):
        cv2.putText(
            panel,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if pose is None:
        put("PnP: NO TARGET", 30, (0, 0, 255))
        put("(detect two light bars)", 60, (150, 150, 150))
        return panel

    if not pose.valid:
        put("PnP: INVALID", 30, (0, 0, 255))
        put(f"Reproj Err: {pose.reprojection_error:.2f} px", 60, (150, 150, 150))
        return panel

    X, Y, Z = pose.position

    put("Position (m):", 30, (0, 255, 0))
    put(f"  X = {X:.3f}", 60)
    put(f"  Y = {Y:.3f}", 90)
    put(f"  Z = {Z:.3f}", 120)

    put("Displacement (m):", 170, (0, 255, 255))

    if init_position is None:
        put("  press r to record init", 200, (150, 150, 150))
    else:
        dX, dY, dZ = pose.position - init_position
        distance = float(np.linalg.norm(pose.position - init_position))
        put(f"  dX = {dX:.3f}", 200)
        put(f"  dY = {dY:.3f}", 230)
        put(f"  dZ = {dZ:.3f}", 260)
        put(f"  Distance = {distance:.3f}", 290)

    return panel


# ==================================================
# 主流程
# ==================================================

def main():

    camera = Camera()
    init_position = None

    while True:

        frame = camera.read()
        if frame is None:
            break

        result = frame.copy()


        # =========================
        # 1. 检测灯带
        # =========================

        candidates, mask = detect_light_bars(frame, mode="white")

        draw_light_bars(result, candidates)


        # =========================
        # 2. 构建目标 + 位姿解算
        # =========================

        target = match_target(candidates)

        pose = None
        if target is not None:
            pose = solve_pose(
                target.image_points,
                camera.camera_matrix,
                camera.dist_coeffs
            )


        # =========================
        # 3. 状态文本
        # =========================

        if pose is None:
            status_text = "PnP: NO TARGET"
            status_color = (0, 0, 255)
        elif pose.valid:
            status_text = (
                f"PnP: VALID   Reproj Err: {pose.reprojection_error:.2f} px"
            )
            status_color = (0, 255, 0)
        else:
            status_text = (
                f"PnP: INVALID   Reproj Err: {pose.reprojection_error:.2f} px"
            )
            status_color = (0, 0, 255)

        cv2.putText(
            result,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )


        # =========================
        # 4. 信息面板（位置 + 位移，实时刷新）
        # =========================

        panel = render_info_panel(pose, init_position)


        # =========================
        # 5. 显示
        # =========================

        cv2.imshow("Camera", result)
        cv2.imshow("Mask", mask)
        cv2.imshow("Info", panel)


        # =========================
        # 6. 按键处理
        # =========================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") and pose is not None and pose.valid:
            init_position = pose.position.copy()

        if key == 27:
            break


    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
