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
