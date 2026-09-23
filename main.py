import cv2

from camera_intrinsics import get_intrinsics
from camera import Camera
from image_process import process_image


def main():

    # ==================================================
    # 1. 获取相机内参
    # ==================================================

    intrinsics = get_intrinsics()


    # ==================================================
    # 2. 创建相机
    # ==================================================

    camera = Camera()


    # ==================================================
    # 3. 实时处理
    # ==================================================

    try:

        while True:

            # 获取当前画面
            frame = camera.get_frame()

            if frame is None:
                continue


            # 图像处理
            result, mask, target_geometry = process_image(
                frame
            )


            # 显示
            cv2.imshow(
                "RealSense RGB",
                result
            )

            cv2.imshow(
                "Mask",
                mask
            )


            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:

        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()