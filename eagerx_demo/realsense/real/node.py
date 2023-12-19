from typing import Optional, List, Dict
import pyrealsense2 as rs
import numpy as np
from threading import Thread

# IMPORT EAGERX
from eagerx import Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class RealSenseSensor(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        stream_rate: int = 30,
        color: Optional[str] = "cyan",
        mode: str = "rgb",
        render_shape: List[int] = None,
    ):
        """A spec to create a RealSenseSensor node that provides images that can be used for perception and/or rendering.

        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param color: Specifies the color of logged messages & node color in the GUI.
        :param mode: Available: `rgb`, `bgr`, `rgbd`, `bgrd`, `bgr` and `rgb`.
        :param render_shape: The shape of the produced images [height, width].
        :return: NodeSpec
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, color=color)
        spec.config.outputs = ["color", "depth"] if "d" in mode else ["color"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        spec.config.stream_rate = stream_rate
        spec.config.render_shape = render_shape if isinstance(render_shape, list) else [480, 680]

        # color
        channels = 3
        color_shape = (spec.config.render_shape[0], spec.config.render_shape[1], channels)
        depth_shape = (spec.config.render_shape[0], spec.config.render_shape[1])
        spec.outputs.color.space = Space(low=0, high=255, shape=color_shape, dtype="uint8")
        spec.outputs.depth.space = Space(low=0, high=20, shape=depth_shape, dtype="float32")
        return spec

    def initialize(self, spec: NodeSpec, simulator: Dict):
        """Initializes the camera sensor according to the spec."""
        self.mode = spec.config.mode
        self.height, self.width = spec.config.render_shape
        self.pipeline = rs.pipeline()
        config = rs.config()
        color_format = rs.format.rgb8 if "rgb" in self.mode else rs.format.bgr8
        config.enable_stream(rs.stream.color, self.width, self.height, color_format, spec.config.stream_rate)
        if "d" in self.mode:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, spec.config.stream_rate)
        self.pipeline.start(config)
        if "d" in self.mode:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
        self.color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth_image = np.zeros((self.height, self.width), dtype=np.float32)
        self.thread = Thread(target=self._image_callback, daemon=True)
        self.thread.start()

    @register.states()
    def reset(self):
        while np.max(self.color_image) == 0 or np.max(self.depth_image) == 0:
            pass

    def _image_callback(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                if "d" in self.mode:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                self.color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                if "d" in self.mode:
                    depth_frame = frames.get_depth_frame()
                    self.depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32) / 10000.0
        except Exception as e:
            print(e)
            self.pipeline.stop()

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(color=Space(dtype="uint8"), depth=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg = None):
        """Produces a camera sensor measurement called `color`.

        The measurements are published at the specified rate * real_time_factor.

        Input `tick` ensures that this node is I/O synchronized with the simulator."""
        msg = dict(color=self.color_image) if "d" not in self.mode else dict(color=self.color_image, depth=self.depth_image)
        return msg
