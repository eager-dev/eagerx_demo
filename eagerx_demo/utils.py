import numpy as np


def string_to_uint8(s):
    # Ensure the string is at most 280 characters
    s = s[:280]
    # Pad the string with null bytes if it is shorter than 280 characters
    s_padded = s.ljust(280, "\0")
    # Convert the string to a numpy array of uint8
    arr = np.frombuffer(s_padded.encode("utf-8"), dtype=np.uint8)
    return arr


def uint8_to_string(arr):
    # Convert the numpy array of uint8 back to bytes
    bytes_str = arr.tobytes()
    # Decode the bytes to a string, and remove any trailing null bytes
    s = bytes_str.decode("utf-8").rstrip("\0")
    return s


def cam_config_to_cam_spec(cam_config):
    cam_spec = []
    for cam in cam_config:
        cam_dict = {}
        for key, value in cam.items():
            cam_dict[key] = list(value) if isinstance(value, tuple) else value
        cam_spec.append(cam_dict)
    return cam_spec


def cam_spec_to_cam_config(cam_spec):
    cam_config = []
    for cam in cam_spec:
        cam_dict = {}
        for key, value in cam.items():
            cam_dict[key] = tuple(value) if isinstance(value, list) else value
        cam_config.append(cam_dict)
    return cam_config