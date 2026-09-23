import pyrealsense2 as rs


def get_intrinsics():

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

    return intrinsics