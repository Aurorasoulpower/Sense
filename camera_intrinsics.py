import numpy as np
import pyrealsense2 as rs


def get_camera_intrinsics():
    """
    获取 RGB 相机内参和畸变参数。

    返回：
        K : np.ndarray (3, 3)
            相机内参矩阵

        dist : np.ndarray
            畸变参数
    """

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

    K = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32
    )

    dist = np.asarray(
        intrinsics.coeffs,
        dtype=np.float32
    )

    pipeline.stop()

    return K, dist
