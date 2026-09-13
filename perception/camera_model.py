"""Pinhole camera model for MuJoCo rendered observations.

MuJoCo cameras look along the **-z** axis of their frame (+x right, +y up)
with a vertical field of view (``model.cam_fovy``, degrees). This module
wraps that convention into a small projection utility used by the
detector to move between world coordinates and image pixels:

* ``project``  — world point -> pixel (u, v) + optical depth.
* ``backproject`` — pixel + depth -> world point.

Only NumPy is used; the matrices come from the live ``MjData`` so moving
cameras (wrist mounts) are supported.
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
# (mujoco's C extension ships incomplete stubs; cam_fovy / cam_xpos /
#  cam_xmat exist at runtime — verified by tests/test_perception.py.)

from __future__ import annotations

import numpy as np


class MuJoCoCamera:
    """Pinhole model of one named MuJoCo camera at the current state."""

    def __init__(self, model, data, name: str, height: int, width: int):
        self.model = model
        self.data = data
        self.name = name
        self.height = height
        self.width = width
        self.cam_id = model.cam(name).id if hasattr(model, "cam") else \
            _cam_id(model, name)
        fovy_rad = np.deg2rad(model.cam_fovy[self.cam_id])
        # square pixels: focal length identical in u and v
        self.f = 0.5 * height / np.tan(0.5 * fovy_rad)
        self.c_u = 0.5 * width
        self.c_v = 0.5 * height

    # ------------------------------------------------------------------ #
    def _pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position + rotation (camera->world) at the current step."""
        pos = np.asarray(self.data.cam_xpos[self.cam_id], dtype=float)
        rot = np.asarray(self.data.cam_xmat[self.cam_id],
                         dtype=float).reshape(3, 3)
        return pos, rot

    def in_front(self, points_world: np.ndarray) -> np.ndarray:
        """Boolean mask of points in front of the camera (>1 cm)."""
        pos, rot = self._pose()
        rel = np.atleast_2d(points_world) - pos
        z_cam = rel @ rot[:, 2]                      # camera -z is forward
        return z_cam < -0.01

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World (N,3) -> pixel (N,2) [u, v] and optical depth (N,)."""
        pos, rot = self._pose()
        rel = np.atleast_2d(points_world) - pos
        # camera frame: x right, y up, z backward (looks along -z)
        x_cam = rel @ rot[:, 0]
        y_cam = rel @ rot[:, 1]
        z_cam = rel @ rot[:, 2]
        depth = -z_cam
        u = self.c_u + self.f * x_cam / depth
        v = self.c_v - self.f * y_cam / depth
        return np.stack([u, v], axis=-1), depth

    def backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """Pixel (u, v) + optical depth -> world point (3,)."""
        pos, rot = self._pose()
        x_cam = (u - self.c_u) * depth / self.f
        y_cam = (self.c_v - v) * depth / self.f
        p_cam = np.array([x_cam, y_cam, -depth])
        return pos + rot @ p_cam


def _cam_id(model, name: str) -> int:
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
