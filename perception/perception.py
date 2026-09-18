"""Perception module: cameras -> world-frame object estimates.

``PerceptionModule`` owns the RGB-D observation pipeline:

* renders the task cameras (overhead for tableware, drawer close-up for
  utensils + handle) at a fixed rate (every ``update_every`` control
  steps — real systems run vision slower than control),
* runs :class:`~perception.detector.ObjectDetector` on each view,
* fuses estimates with an exponentially-weighted filter + outlier gate,
* estimates the drawer slide from the detected gold handle post.

The module is the single swap point for a learned perception backbone
(e.g. an OpenVINO-optimized segmentation model): replace ``update`` with
a network call that fills the same ``self.est`` table.

All queries are world-frame and use ONLY camera observations plus arm
proprioception (gripper self-image suppression) — never privileged
simulator state.
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import numpy as np

from perception.camera_model import MuJoCoCamera
from perception.detector import Detection, ObjectDetector

# object -> camera assignment (world.xml cameras)
CAMERA_ASSIGNMENT = {
    "overhead":   ["plate", "mug", "bottle"],
    "drawer_cam": ["spoon", "fork", "__handle"],
}

# drawer geometry (must match envs.dinner_table_env): world x of the
# handle post centre when the drawer is closed
HANDLE_X_CLOSED = -0.398
DRAWER_TRAVEL = 0.26


class PerceptionModule:
    """Multi-camera RGB-D perception with temporal filtering."""

    def __init__(self, env, resolution: int = 448, alpha: float = 0.35,
                 gate: float = 0.05, update_every: int = 10):
        self.env = env
        self.res = resolution
        self.alpha = alpha          # EMA weight of new evidence
        self.gate = gate            # m; larger jumps need high confidence
        self.update_every = update_every
        self.detector = ObjectDetector(resolution=resolution)
        self._cams: dict[str, MuJoCoCamera] = {}
        self._renderers: dict[str, object] = {}
        self._step = 0
        # per-target fused state
        self.est: dict[str, np.ndarray] = {}
        self.yaw: dict[str, float | None] = {}
        self.conf: dict[str, float] = {}
        self.last_detections: dict[str, Detection] = {}
        self.drawer_q: float | None = None      # estimated drawer slide

    # ------------------------------------------------------------------ #
    def _camera(self, name: str):
        import mujoco
        cam = self._cams.get(name)
        if cam is None:
            cam = MuJoCoCamera(self.env.model, self.env.data, name,
                               self.res, self.res)
            self._cams[name] = cam
        else:
            cam.data = self.env.data            # always current state
        return cam

    def _render(self, cam_name: str) -> tuple[np.ndarray, np.ndarray]:
        import mujoco
        r = self._renderers.get(cam_name)
        if r is None:
            r = mujoco.Renderer(self.env.model, height=self.res,
                                width=self.res)
            self._renderers[cam_name] = r
        r.update_scene(self.env.data, camera=cam_name)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(self.env.data, camera=cam_name)
        depth = np.asarray(r.render(), dtype=np.float64)
        r.disable_depth_rendering()
        return rgb, depth

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear temporal state (new episode)."""
        self.est.clear()
        self.yaw.clear()
        self.conf.clear()
        self.last_detections.clear()
        self.drawer_q = None
        self._step = 0

    def close(self) -> None:
        """Release MuJoCo renderers/GL contexts owned by perception."""
        for renderer in self._renderers.values():
            close = getattr(renderer, "close", None)
            if close is not None:
                close()
        self._renderers.clear()
        self._cams.clear()

    def update(self, drawer_shift: float | None = None, force: bool = False) \
            -> dict[str, Detection]:
        """Run detection on all assigned cameras (rate-limited).

        ``drawer_shift`` overrides the drawer-slide prior used for the
        tray search volumes (defaults to the module's own estimate).
        """
        self._step += 1
        if not force and (self._step % self.update_every) != 0:
            return self.last_detections

        shift = self.drawer_q if drawer_shift is None else drawer_shift
        shift = 0.0 if shift is None else float(shift)

        # proprioceptive self-image: project gripper sites into each camera
        grips = {c: [] for c in CAMERA_ASSIGNMENT}
        for arm in ("A", "B"):
            gp = self.env.gripper_site_pos(arm)
            for cam_name, cam in ((n, self._camera(n))
                                  for n in CAMERA_ASSIGNMENT):
                uv, d = cam.project(gp[None, :])
                if cam.in_front(gp[None, :])[0] and 0 <= uv[0, 0] < self.res \
                        and 0 <= uv[0, 1] < self.res:
                    grips[cam_name].append((float(uv[0, 0]), float(uv[0, 1])))

        prior = {k: v for k, v in self.est.items()}
        all_dets: dict[str, Detection] = {}
        for cam_name, targets in CAMERA_ASSIGNMENT.items():
            cam = self._camera(cam_name)
            rgb, depth = self._render(cam_name)
            dets = self.detector.detect(rgb, depth, cam, targets,
                                        tray_shift=shift,
                                        gripper_px=grips[cam_name],
                                        prior=prior)
            all_dets.update(dets)

        for name, det in all_dets.items():
            self._fuse(name, det)

        # drawer slide from the handle estimate
        h = all_dets.get("__handle")
        if h is not None and h.valid():
            q = float(np.clip(h.pos[0] - HANDLE_X_CLOSED, 0.0, DRAWER_TRAVEL))
            self.drawer_q = q if self.drawer_q is None \
                else (1 - self.alpha) * self.drawer_q + self.alpha * q
        return self.last_detections

    # ------------------------------------------------------------------ #
    def _fuse(self, name: str, det: Detection) -> None:
        self.last_detections[name] = det
        if not det.valid():
            self.conf[name] = self.conf.get(name, 0.0) * 0.95
            return
        prev = self.est.get(name)
        if prev is None:
            self.est[name] = det.pos.copy()
        else:
            jump = float(np.linalg.norm(det.pos - prev))
            if jump < self.gate or det.conf > 0.8:
                a = self.alpha if jump < self.gate else 1.0
                self.est[name] = (1 - a) * prev + a * det.pos
            else:
                return                       # rejected outlier
        self.conf[name] = max(self.conf.get(name, 0.0) * 0.9, det.conf)
        if det.yaw is not None:
            self.yaw[name] = det.yaw

    # ------------------------------------------------------------------ #
    # Queries (used by the orchestrator and, via the env pose provider,
    # by the skill layer)
    # ------------------------------------------------------------------ #
    def object_pos(self, name: str) -> np.ndarray | None:
        return self.est.get(name)

    def object_yaw(self, name: str) -> float | None:
        return self.yaw.get(name)

    def confidence(self, name: str) -> float:
        return self.conf.get(name, 0.0)

    def handle_pos(self) -> np.ndarray | None:
        return self.est.get("__handle")

    def drawer_qpos(self) -> float | None:
        return self.drawer_q

    def snapshot(self) -> dict:
        return {
            "est": {k: v.copy() for k, v in self.est.items()},
            "yaw": dict(self.yaw),
            "conf": dict(self.conf),
            "drawer_q": self.drawer_q,
        }

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
