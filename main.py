import cv2
import numpy as np

from camera_intrinsics import get_camera_intrinsics
from camera import Camera
from image_process import process_image
from pose_solver import solve_pose, get_position, position_to_meter


def main():

    # =========================
    # 1. 获取相机内参
    # =========================

    K, dist = get_camera_intrinsics()


    # =========================
    # 2. 创建相机
    # =========================

    camera = Camera()

    init_position = None


    # =========================
    # 3. 实时处理
    # =========================

    while True:

        # 获取一帧图像
        frame = camera.read()

        if frame is None:
            break


        # =========================
        # 4. 图像处理
        # =========================

        result, mask, target_geometry = process_image(frame)

        position_available = False


        if target_geometry is not None:

            led1 = target_geometry["led1"]
            led2 = target_geometry["led2"]


            # =========================
            # 5. 获取 8 个图像角点
            # =========================

            led1_points = led1["box"]
            led2_points = led2["box"]

            image_points = np.vstack(
                (led1_points, led2_points)
            ).astype(np.float32)


            # =========================
            # 6. PnP 空间位置解算
            # =========================

            success, rvec, tvec = solve_pose(
                image_points,
                K,
                dist
            )


            if success:

                # tvec 单位：mm
                position_mm = get_position(tvec)

                # mm → m
                position = position_to_meter(
                    position_mm
                )

                X, Y, Z = position
                position_available = True


                # =========================
                # 7. 计算位移
                # =========================

                if init_position is not None:

                    delta_position = (
                        position - init_position
                    )

                    dX, dY, dZ = delta_position

                    distance = float(
                        np.linalg.norm(
                            delta_position
                        )
                    )

                    print(
                        f"位置: "
                        f"X={X:.3f} m, "
                        f"Y={Y:.3f} m, "
                        f"Z={Z:.3f} m | "
                        f"位移: "
                        f"dX={dX:.3f} m, "
                        f"dY={dY:.3f} m, "
                        f"dZ={dZ:.3f} m, "
                        f"Distance={distance:.3f} m"
                    )

                else:

                    print(
                        f"目标位置: "
                        f"X={X:.3f} m, "
                        f"Y={Y:.3f} m, "
                        f"Z={Z:.3f} m"
                    )


        # =========================
        # 8. 显示图像
        # =========================

        cv2.imshow(
            "Camera",
            result
        )


        # =========================
        # 9. 按键处理（每帧只取一次）
        # =========================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") and position_available:
            init_position = position.copy()
            print("已记录初始位置")

        if key == 27:
            break


    # =========================
    # 10. 释放资源
    # =========================

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()