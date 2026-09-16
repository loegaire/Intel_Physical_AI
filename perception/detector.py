"""Camera-based object detection for the dinner-table scene.

The detector is honest computer vision: it consumes only rendered RGB-D
observations (no privileged simulator state) and produces world-frame
object estimates by

1. thresholding a per-object colour class in HSV (the class priors are
   the *nominal* object colours — a real system would learn them),
2. intersecting with a projected world-frame search volume (task prior:
   ``counter`` for tableware, the drawer tray interior for utensils, the
   drawer front for the handle),
3. optionally intersecting with a *height above the local ground plane*
   mask from the depth channel (separates silver utensils from the tray
   floor they lie on), and
4. picking the best connected component and back-projecting its
   pixel centroid + median depth through the pinhole model.

Gripper self-image is suppressed by masking discs around each arm's
proprioceptively-known gripper position (a real robot knows where its
own end-effectors are from kinematics).

Domain randomization moves object colours by +-0.05 RGB per channel, so
all colour windows carry margin. Yaw of elongated objects (utensils) is
estimated from the principal axis of the component's pixel set.
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.camera_model import MuJoCoCamera


@dataclass
class Detection:
    name: str
    pos: np.ndarray            # world-frame body-centre estimate (3,)
    yaw: float | None = None   # estimated yaw about +z (rad) or None
    conf: float = 0.0          # 0..1 detection confidence
    n_pixels: int = 0

    def valid(self) -> bool:
        return self.conf > 0.0


@dataclass
class ObjectClass:
    """Colour + geometry prior for one object type."""
    hsv_lo: tuple[int, int, int]
    hsv_hi: tuple[int, int, int]
    z_offset: float            # backprojected surface -> body centre (dz)
    size_px: tuple[int, int]   # acceptable component area range at 448 px
    floor_z: float | None = None   # world z of the local ground plane
    height_lo: float = 0.003   # m above floor_z (surface z - floor_z)
    height_hi: float = 0.05
    elongated: bool = False    # estimate yaw from the pixel principal axis


# HSV windows calibrated on RENDERED pixels (the scene lighting saturates
# bright surfaces, so nominal albedo is not a reliable guide):
#   plate/placemat render V=255 white  -> plate needs the height mask
#   mug renders hue ~18, sat ~135 washed to ~55 by lighting (arm yellow
#     is hue ~30, counter wood hue ~30 at sat ~39)
#   bottle renders hue ~73, sat ~160
#   utensils render hue ~110, sat ~23  (tray floor is hue ~106, sat ~12;
#     only the world-z height mask separates shaft from tray)
#   handle renders hue ~8-10, sat ~12-60 depending on lighting
# floor_z values match envs/world.xml geometry: countertop top z=0.75,
# placemat top z=0.7535, drawer tray floor top z=0.710.
CLASSES: dict[str, ObjectClass] = {
    "mug":    ObjectClass(hsv_lo=(8, 55, 60),     hsv_hi=(26, 255, 255),
                          z_offset=0.027, size_px=(25, 4000),
                          floor_z=0.750, height_lo=0.015, height_hi=0.090),
    "bottle": ObjectClass(hsv_lo=(60, 100, 40),   hsv_hi=(95, 255, 255),
                          z_offset=0.155, size_px=(40, 20000),
                          floor_z=0.750, height_lo=0.030, height_hi=0.28),
    "plate":  ObjectClass(hsv_lo=(0, 0, 200),     hsv_hi=(180, 40, 255),
                          z_offset=0.010, size_px=(400, 30000),
                          floor_z=0.750, height_lo=0.005, height_hi=0.045),
    "spoon":  ObjectClass(hsv_lo=(85, 0, 140),    hsv_hi=(130, 60, 255),
                          z_offset=0.004, size_px=(8, 4000),
                          floor_z=0.710, height_lo=0.002, height_hi=0.030,
                          elongated=True),
    "fork":   ObjectClass(hsv_lo=(85, 0, 140),    hsv_hi=(130, 60, 255),
                          z_offset=0.004, size_px=(8, 4000),
                          floor_z=0.710, height_lo=0.002, height_hi=0.030,
                          elongated=True),
    # gold drawer handle post (~2 x 6 cm) on the drawer front
    "__handle": ObjectClass(hsv_lo=(3, 0, 50),    hsv_hi=(16, 255, 225),
                            z_offset=0.0, size_px=(4, 3000)),
}


# --------------------------------------------------------------------------- #
class ObjectDetector:
    """RGB-D object detector over MuJoCo rendered observations."""

    # world-frame search volumes per target (x0,x1, y0,y1, z0,z1);
    # 'tray' volumes are shifted by the drawer slide estimate at query time
    SEARCH = {
        "plate":  (-0.20, 0.55, -0.36, 0.36, 0.70, 0.95),
        "mug":    (-0.20, 0.55, -0.36, 0.36, 0.70, 0.95),
        "bottle": (-0.20, 0.55, -0.36, 0.36, 0.70, 1.05),
        "spoon":  ("tray",),
        "fork":   ("tray",),
        # the handle travels +x by the drawer travel (closed x = -0.54)
        "__handle": (-0.55, -0.05, -0.31, -0.05, 0.68, 0.86),
    }

    GRIP_EXCLUDE_R = 0.055     # m self-image suppression radius per gripper

    def __init__(self, resolution: int = 448):
        self.res = resolution

    # ------------------------------------------------------------------ #
    def detect(self, rgb: np.ndarray, depth: np.ndarray,
               camera: MuJoCoCamera, targets: list[str],
               tray_shift: float = 0.0,
               gripper_px: list[tuple[float, float]] | None = None,
               prior: dict[str, np.ndarray] | None = None) \
            -> dict[str, Detection]:
        """Detect ``targets`` in one RGB-D observation.

        ``tray_shift`` — current drawer slide (m) applied to tray volumes.
        ``gripper_px`` — projected gripper points to mask out (self-image).
        ``prior``      — optional per-object last-known world position;
                         ties between components are broken by prior
                         proximity.
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        out: dict[str, Detection] = {}
        for name in targets:
            vol = self._volume(name, tray_shift)
            det = self._detect_one(name, hsv, depth, camera, vol,
                                   gripper_px or [],
                                   (prior or {}).get(name))
            out[name] = det
        return out

    # ------------------------------------------------------------------ #
    def _volume(self, name: str, tray_shift: float) -> np.ndarray:
        v = self.SEARCH[name]
        if v[0] == "tray":
            # tray interior, shrunk by the wall thickness; x follows the
            # estimated drawer slide (closed interior x [-0.795,-0.555])
            return np.array([-0.795 + tray_shift, -0.555 + tray_shift,
                             -0.255, -0.105, 0.705, 0.75])
        return np.array(v, dtype=float)

    def _detect_one(self, name: str, hsv: np.ndarray, depth: np.ndarray,
                    camera: MuJoCoCamera, vol: np.ndarray,
                    gripper_px: list[tuple[float, float]],
                    prior: np.ndarray | None) -> Detection:
        cls = CLASSES[name]
        res = self.res

        # 1) colour class mask
        mask = cv2.inRange(hsv, np.array(cls.hsv_lo, np.uint8),
                           np.array(cls.hsv_hi, np.uint8)) > 0

        # 2) projected world search volume
        volume_mask = self._volume_mask(camera, vol, depth)
        mask &= volume_mask

        # 3) self-image suppression
        if gripper_px:
            yy, xx = np.mgrid[0:res, 0:res]
            for (gu, gv) in gripper_px:
                r_px = self.GRIP_EXCLUDE_R * self.res * 2.0 / (
                    camera.f * 0.9 + 1e-9)          # ~5.5 cm at 0.9 m
                mask &= ((xx - gu) ** 2 + (yy - gv) ** 2) > r_px ** 2

        # 4) world-z height mask: keep pixels whose back-projected
        # surface point stands height_lo..height_hi above floor_z
        # (exact w.r.t. the camera angle — no image-space heuristics)
        if cls.floor_z is not None:
            mask &= self._height_mask(depth, mask, camera, cls.floor_z,
                                      cls.height_lo, cls.height_hi)

        # 5) connected components -> best candidate
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return Detection(name, np.zeros(3), conf=0.0)
        best, best_score = None, -1.0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (cls.size_px[0] <= area <= cls.size_px[1]):
                continue
            w_px = stats[i, cv2.CC_STAT_WIDTH]
            h_px = stats[i, cv2.CC_STAT_HEIGHT]
            if max(w_px, h_px) > 0.55 * res:        # implausibly huge blob
                continue
            cx, cy = cents[i]                        # OpenCV returns (x, y)
            score = float(area)
            if prior is not None:
                prior_px, _ = camera.project(prior[None, :])
                d = np.linalg.norm(cents[i] - prior_px[0])
                score *= float(np.exp(-d / (0.12 * res)))   # prior affinity
            if score > best_score:
                best_score, best = score, (i, cx, cy, area, w_px, h_px)
        if best is None:
            return Detection(name, np.zeros(3), conf=0.0)
        i, cx, cy, area, w_px, h_px = best

        # 6) metric pose: median depth over the component -> backproject
        comp = labels == i
        d_med = float(np.median(depth[comp]))
        if not np.isfinite(d_med) or d_med <= 0.02:
            return Detection(name, np.zeros(3), conf=0.0)
        p_surf = camera.backproject(cx, cy, d_med)
        pos = p_surf.copy()
        pos[2] -= cls.z_offset

        yaw = None
        if cls.elongated and area >= 2 * cls.size_px[0]:
            yaw = self._principal_yaw(comp, d_med, camera)

        conf = float(np.clip(area / cls.size_px[1], 0.15, 1.0))
        return Detection(name, pos, yaw=yaw, conf=conf, n_pixels=area)

    # ------------------------------------------------------------------ #
    def _volume_mask(self, camera: MuJoCoCamera, vol: np.ndarray,
                     depth: np.ndarray) -> np.ndarray:
        """Rasterise a world-frame axis-aligned box into a pixel mask.

        Conservative: projects the box's 8 corners and floods the convex
        hull, then rejects pixels whose depth is outside the box's z-range
        mapped through the camera (approximate, keeps masks tight).
        """
        res = self.res
        xs, ys, zs = vol[0:2], vol[2:4], vol[4:6]
        corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        uv, d = camera.project(corners)
        if not camera.in_front(corners).any():
            return np.zeros((res, res), bool)
        u = np.clip(uv[:, 0], 0, res - 1).astype(int)
        v = np.clip(uv[:, 1], 0, res - 1).astype(int)
        pts = np.stack([u, v], axis=1).astype(np.int32)
        # fill the convex hull IN HULL ORDER (corner order zigzags and
        # would self-intersect, blanking parts of the volume)
        hull = cv2.convexHull(pts)
        canvas = np.zeros((res, res), np.uint8)
        cv2.fillConvexPoly(canvas, hull, 1)
        m = canvas.astype(bool)
        # depth gate: keep pixels whose rendered surface could belong to
        # the volume (the surface may be the box top OR its interior)
        dmin, dmax = float(d.min()), float(d.max())
        m &= (depth > max(dmin - 0.10, 0.02)) & (depth < dmax + 0.10)
        return m

    def _height_mask(self, depth: np.ndarray, color_mask: np.ndarray,
                     camera: MuJoCoCamera, floor_z: float,
                     h_lo: float, h_hi: float) -> np.ndarray:
        """Keep candidates whose surface height above ``floor_z`` is in
        [h_lo, h_hi] — computed by back-projecting each masked pixel with
        its rendered depth and comparing the world z (exact at any
        camera angle, robust to depth gradients across the image)."""
        ys, xs = np.nonzero(color_mask)
        if len(xs) == 0:
            return color_mask
        d = depth[ys, xs]
        ok = np.isfinite(d) & (d > 0.02)
        zs = np.full(len(xs), -np.inf)
        if ok.any():
            pos, rot = camera._pose()
            xc = (xs[ok] - camera.c_u) * d[ok] / camera.f
            yc = (camera.c_v - ys[ok]) * d[ok] / camera.f
            pc = np.stack([xc, yc, -d[ok]], axis=1)
            zw = pc @ rot[:, 2] + pos[2]
            zs[ok] = zw
        keep = (zs - floor_z > h_lo) & (zs - floor_z < h_hi)
        out = np.zeros_like(color_mask)
        out[ys[keep], xs[keep]] = True
        return out

    def _principal_yaw(self, comp: np.ndarray, d_med: float,
                       camera: MuJoCoCamera) -> float | None:
        """Object yaw from the component's pixel principal axis."""
        ys, xs = np.nonzero(comp)
        if len(xs) < 6:
            return None
        pts = np.stack([xs, ys], axis=1).astype(float)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        w, vecs = np.linalg.eigh(cov)
        if w[-1] < 1e-8 or w[-1] / max(w[0], 1e-8) < 4.0:
            return None                           # not elongated enough
        axis_px = vecs[:, -1]
        ext = float(np.sqrt(w[-1]))
        p0 = camera.backproject(mean[0] - axis_px[0] * ext,
                                mean[1] - axis_px[1] * ext, d_med)
        p1 = camera.backproject(mean[0] + axis_px[0] * ext,
                                mean[1] + axis_px[1] * ext, d_med)
        v = p1[:2] - p0[:2]
        n = np.linalg.norm(v)
        if n < 0.02:                              # shorter than a utensil
            return None
        return float(np.arctan2(v[1], v[0]))
