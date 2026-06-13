import pyrealsense2 as rs
import numpy as np


class RealSenseOptimize:

    def __init__(
        self,
        width=848,
        height=480,
        fps=15
    ):

        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.cfg.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps
        )

        self.cfg.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps
        )

        self.profile = self.pipe.start(self.cfg)

        self.align = rs.align(rs.stream.color)

        self.depth_sensor = (
            self.profile
            .get_device()
            .first_depth_sensor()
        )

        self.original_preset = (
            self.depth_sensor.get_option(
                rs.option.visual_preset
            )
        )

        # High Density
        self.depth_sensor.set_option(
            rs.option.visual_preset,
            4
        )

        print("Using preset: High Density (4)")

        self.spatial = rs.spatial_filter()

        self.spatial.set_option(
            rs.option.filter_magnitude,
            5
        )

        self.spatial.set_option(
            rs.option.filter_smooth_alpha,
            0.5
        )

        self.spatial.set_option(
            rs.option.filter_smooth_delta,
            50
        )

        # Optional
        self.temporal = rs.temporal_filter()

        self.temporal.set_option(
            rs.option.filter_smooth_alpha,
            0.4
        )

        self.temporal.set_option(
            rs.option.filter_smooth_delta,
            20
        )

    def get_frames(self):

        frames = self.pipe.wait_for_frames()

        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Spatial filter
        depth_frame = self.spatial.process(
            depth_frame
        )

        # Nếu muốn bật temporal sau này
        depth_frame = self.temporal.process(depth_frame)

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )

        return color_image, depth_image

    def get_raw_frames(self):

        frames = self.pipe.wait_for_frames()

        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )

        return color_image, depth_image

    def stop(self):

        try:

            self.depth_sensor.set_option(
                rs.option.visual_preset,
                1
            )

            print(
                f"Restored preset: {1}"
            )

        except Exception as e:
            print(e)

        self.pipe.stop()