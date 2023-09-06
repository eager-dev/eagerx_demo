import cv2
from copy import deepcopy
import numpy as np
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import ctypes


class DemonstrationWindow(object):
    def __init__(self, img):
        self.img_original = deepcopy(cv2.rotate(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
        self.img_original = np.asarray(cv2.flip(self.img_original, 1), dtype="uint8")
        self.img = deepcopy(self.img_original)
        self.points = []

    def demonstrate(self):
        window_name = "Demonstration"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            cv2.imshow(window_name, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k in [27, 10, 13]:
                break
            elif k == 8:
                if len(self.points) > 0:
                    self.img = deepcopy(self.img_original)
                    self.points.pop(-1)
                    for idx, point in enumerate(self.points):
                        cv2.circle(self.img, (point[0], point[1]), 2, (255, 0, 0), -1)
                        if idx in [1, 3]:
                            cv2.line(self.img, tuple(self.points[idx - 1]), tuple(self.points[idx]), (255, 255, 255), 2)
        cv2.destroyWindow(window_name)
        return self.points

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
            self.points.append([x, y])
            if len(self.points) in [2, 4]:
                cv2.line(
                    self.img,
                    tuple(self.points[-2]),
                    tuple(self.points[-1]),
                    (255, 255, 255),
                    2,
                )

def demonstrate(img):
    shared_array_base = multiprocessing.Array(ctypes.c_int32, 4 * 2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_points = shared_array.reshape(4, 2)
    shape = img.shape
    shared_img = np.ndarray(shape, dtype="uint8", buffer=RawArray(ctypes.c_uint8, img.reshape(-1)))
    args = (shared_img, shape, shared_points)
    p = multiprocessing.Process(target=_demonstrate, args=args)
    p.start()
    p.join()
    return shared_points.tolist()


def _demonstrate(img, shape, shared_points):
    print("Demonstrating...")
    img_original = img.view(dtype="uint8").reshape(shape)
    demo_window = DemonstrationWindow(img_original)
    demo_points = demo_window.demonstrate()
    print("Demonstration done.")
    for idx, point in enumerate(demo_points):
        shared_points[idx][0] = point[0]
        shared_points[idx][1] = point[1]
