"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseConfig:
    image_size = (480, 848)
    intrinsics = (431.0718688964844, 0.0, 415.29388427734375, 0.0, 430.6985778808594, 244.10433959960938, 0.0, 0.0, 1.0)
    front_position = (0.609, -0.064, 0.295)
    front_rotation = (0.715, -0.699, -0.005, 0.001)

    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        }
    ]


# class RealSenseD415:
#     """Default configuration with 3 RealSense RGB-D cameras."""

#     # Mimic RealSense D415 RGB-D camera parameters.
#     image_size = (480, 640)
#     intrinsics = (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1)

#     # Set default camera poses.
#     front_position = (1.0, 0, 0.75)
#     front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
#     front_rotation = p.getQuaternionFromEuler(front_rotation)
#     left_position = (0, 0.5, 0.75)
#     left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
#     left_rotation = p.getQuaternionFromEuler(left_rotation)
#     right_position = (0, -0.5, 0.75)
#     right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
#     right_rotation = p.getQuaternionFromEuler(right_rotation)

#     # Default camera configs.
#     CONFIG = [
#         {
#             "image_size": image_size,
#             "intrinsics": intrinsics,
#             "position": front_position,
#             "rotation": front_rotation,
#             "zrange": (0.01, 10.0),
#             "noise": False,
#         },
#         {
#             "image_size": image_size,
#             "intrinsics": intrinsics,
#             "position": left_position,
#             "rotation": left_rotation,
#             "zrange": (0.01, 10.0),
#             "noise": False,
#         },
#         {
#             "image_size": image_size,
#             "intrinsics": intrinsics,
#             "position": right_position,
#             "rotation": right_rotation,
#             "zrange": (0.01, 10.0),
#             "noise": False,
#         },
#     ]


class Oracle:
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)
    position = (0.5, 0, 1000.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (999.7, 1001.0),
            "noise": False,
        }
    ]


class RS200Gazebo:
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (554.3826904296875, 0.0, 320.0, 0.0, 554.3826904296875, 240.0, 0.0, 0.0, 1.0)
    position = (0.5, 0, 1.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, np.pi / 2))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        }
    ]


class KinectFranka:
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424, 512)
    intrinsics = (365.57489013671875, 0.0, 257.5205078125, 0.0, 365.57489013671875, 205.26710510253906, 0.0, 0.0, 1.0)
    position = (1.082, -0.041, 1.027)
    rotation = p.getQuaternionFromEuler((-2.611, 0.010, 1.553))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        }
    ]
