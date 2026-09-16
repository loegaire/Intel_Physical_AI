import cv2
import numpy as np

from perception.detector import CLASSES, ObjectDetector


class IdentityCamera:
    f = 1.0

    def project(self, points):
        return points[:, :2], np.ones(len(points))

    def in_front(self, points):
        return np.ones(len(points), dtype=bool)

    def backproject(self, u, v, depth):
        return np.asarray([u, v, depth], dtype=float)


def test_connected_component_centroid_keeps_opencv_xy_order(monkeypatch):
    detector = ObjectDetector(resolution=20)
    hsv = np.zeros((20, 20, 3), dtype=np.uint8)
    depth = np.ones((20, 20), dtype=float)
    cls = CLASSES["__handle"]
    hsv[7:11, 12:16] = np.asarray(cls.hsv_lo, dtype=np.uint8)
    monkeypatch.setattr(detector, "_volume_mask", lambda *args: np.ones((20, 20), dtype=bool))

    detection = detector._detect_one(
        "__handle", hsv, depth, IdentityCamera(), np.zeros(6), [], None
    )

    # Rectangle centroid is x=13.5, y=8.5. This guards the historical swap.
    assert detection.valid()
    assert detection.pos[0] == 13.5
    assert detection.pos[1] == 8.5
