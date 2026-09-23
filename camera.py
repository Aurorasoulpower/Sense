import pyrealsense2 as rs
import numpy as np


class Camera:

    def __init__(self):

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            30
        )

        self.pipeline.start(self.config)


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
