import pyrealsense2 as rs
import numpy as np


# 各选项的显示名称，用于 print_settings 打印
_OPTION_NAMES = {
    rs.option.exposure: "曝光 exposure",
    rs.option.gain: "增益 gain",
    rs.option.white_balance: "白平衡 white_balance",
}


class Camera:

    def __init__(self, exposure=50, gain=50, white_balance=4600):
        """
        初始化 RealSense 彩色相机，并关闭自动曝光 / 自动白平衡，
        改用固定手动参数，避免强光灯带被自动曝光打到过曝。

        同时读取相机内参与畸变系数，存为：
            camera_matrix : (3, 3) 内参矩阵
            dist_coeffs   : (n,)   畸变系数

        参数：
            exposure       曝光时间，单位 100µs（D400 默认约 156，即 15.6ms）
            gain           传感器增益（D400 默认约 64，越大越亮、噪点越多）
            white_balance  白平衡色温，单位 K（默认 4600）
        """

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            30
        )

        self.profile = self.pipeline.start(self.config)
        self.sensor = self.profile.get_device().first_color_sensor()

        if self.sensor is not None:
            self._disable_auto()
            self.set_option(rs.option.exposure, exposure)
            self.set_option(rs.option.gain, gain)
            self.set_option(rs.option.white_balance, white_balance)

        self.camera_matrix, self.dist_coeffs = self._get_intrinsics()


    def _get_intrinsics(self):
        """从已启动的彩色流中读取内参矩阵和畸变系数。"""

        color_profile = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()

        intrinsics = color_profile.get_intrinsics()

        camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32
        )

        dist_coeffs = np.asarray(
            intrinsics.coeffs,
            dtype=np.float32
        )

        return camera_matrix, dist_coeffs


    def _disable_auto(self):
        """关闭自动曝光和自动白平衡，让手动参数真正生效。"""

        for option in (
            rs.option.enable_auto_exposure,
            rs.option.enable_auto_white_balance,
        ):
            if self.sensor.supports(option):
                self.sensor.set_option(option, 0)


    def set_option(self, option, value):
        """设置相机选项，自动把 value 夹在传感器支持范围内；不支持则忽略。"""

        if self.sensor is None or value is None:
            return

        if not self.sensor.supports(option):
            return

        rng = self.sensor.get_option_range(option)
        value = min(max(value, rng.min), rng.max)

        self.sensor.set_option(option, value)


    def get_option(self, option):
        """读取相机选项当前值；没有传感器或不支持时返回 None。"""

        if self.sensor is None or not self.sensor.supports(option):
            return None

        return self.sensor.get_option(option)


    def print_settings(self):
        """打印曝光 / 增益 / 白平衡的当前值和可用范围，方便调参。"""

        if self.sensor is None:
            print("未找到彩色传感器")
            return

        for option in (rs.option.exposure, rs.option.gain, rs.option.white_balance):
            rng = self.sensor.get_option_range(option)
            name = _OPTION_NAMES[option]
            current = self.sensor.get_option(option)
            print(
                f"{name}: 当前={current:.1f}  "
                f"范围=[{rng.min:.0f}, {rng.max:.0f}]  "
                f"步长={rng.step:.0f}  默认={rng.default:.0f}"
            )


    def read(self):
        """
        读取一帧 BGR 图像。

        返回：
            img : np.ndarray
                获取成功返回图像，失败返回 None
        """

        frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        img = np.asanyarray(
            color_frame.get_data()
        )

        return img


    def release(self):

        self.pipeline.stop()
