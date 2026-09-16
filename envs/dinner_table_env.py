"""Bimanual SO-101 dinner-table manipulation environment (MuJoCo).

Scene layout (world frame, see ``envs/world.xml``):

* A wooden counter (top at z = 0.75) with a sliding drawer on the left.
* Arm A (Menagerie SO-101, prefixed ``A/``) sits at (-0.22, -0.34, 0.75) and
  owns the **drawer zone** (x < -0.30).
* Arm B (prefixed ``B/``) sits at (0.10, 0.30, 0.75) and owns the **placemat
  zone** (x > -0.05); the placemat with plate/fork/spoon/mug slots is on the
  right half of the counter.
* Both arms reach a shared **handoff region** around (-0.08, 0, 0.86).
* The bottle stands behind the placemat; pouring = tilting the bottle above
  the mug (flagged by task-state logic).

Domain randomization (per seed): object positions, yaw, friction, mass,
size scale, colours, and light intensity.

Control: 12 absolute joint-position targets (A then B, 6 joints each) at
50 Hz, integrated at 200 Hz (4 sim steps per control step).
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
# (mujoco's C extension ships incomplete stubs: MjSpec, MjData, mjtGeom,
#  mjtObj, mj_name2id, mj_forward, mj_jacSite, Renderer all exist at
#  runtime — verified in scripts/study_workspace.py and env smoke tests.)

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

# NOTE: mujoco's C-extension ships incomplete type stubs — MjSpec, MjData,
# mj_name2id, mjtGeom, Renderer, mj_forward, mj_jacSite all exist at runtime.
# See the pyright directive at the top of this file for the suppression.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_XML = os.path.join(ROOT, "envs", "world.xml")
# Prefer the hand-tuned primitive model (tracked, no STL meshes needed).
# assets/so101/so101.xml is the Menagerie STL model and only loads when the
# meshes are present locally.
SO101_XML_GEN = os.path.join(ROOT, "assets", "so101", "so101_generated.xml")
SO101_XML_ORIG = os.path.join(ROOT, "assets", "so101", "so101.xml")
SO101_XML = SO101_XML_GEN if os.path.exists(SO101_XML_GEN) else SO101_XML_ORIG

# Drawer geometry (world frame; see envs/world.xml):
# drawer body frame at world (-0.68, -0.18, 0); tray floor top at z = 0.710.
# The cabinet stands LEFT of the countertop (x < -0.50). The tray slides
# +x by qpos (0.26 fully open). CLOSED tray interior x [-0.805,-0.545];
# OPEN tray x [-0.545,-0.285] — the open tray's -x end is left of the
# countertop edge (-0.40) in the clear zone.
DRAWER_UNIT_POS = np.array([-0.68, -0.18, 0.0])
DRAWER_FLOOR_Z = 0.710
DRAWER_TRAVEL = 0.26
# handle centre world (closed) — grasp target for the open-drawer skill
DRAWER_HANDLE_POS = np.array([-0.398, -0.162, 0.758])

# --------------------------------------------------------------------------- #
# Scene constants
# --------------------------------------------------------------------------- #
COUNTERTOP_Z = 0.75
ARMS = ("A", "B")
N_ARM_JOINTS = 6
SIMSTEPS_PER_CTRL = 4           # 200 Hz sim, 50 Hz control

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")

ARM_BASES = {
    "A": {"pos": (-0.22, -0.34, COUNTERTOP_Z), "yaw": math.atan2(0.34, 0.22)},
    "B": {"pos": (0.10, 0.30, COUNTERTOP_Z), "yaw": -1.892546881191539},
}

# World-frame object definitions: (start_xyz, goal_xy or None)
# Utensils start inside the drawer tray (z = tray floor top + item half-height).
DEFAULT_OBJECTS = (
    # plate starts fully on the counter (countertop -x edge at x = -0.10;
    # plate radius 0.105 -> centre must stay at x >= 0.0 to not overhang)
    ("plate",  (0.00, -0.08, COUNTERTOP_Z + 0.008), (0.25, 0.03)),
    ("mug",    (-0.05,  0.14, COUNTERTOP_Z + 0.042), (0.31, -0.10)),
    ("bottle", (0.30,  0.17, COUNTERTOP_Z + 0.075), None),
    # Utensil heads slightly overhang the slotted front bar, leaving a
    # reachable top-down grasp point after the drawer opens.
    ("spoon",  (-0.580, -0.17, DRAWER_FLOOR_Z + 0.020), (0.11, 0.10)),
    ("fork",   (-0.580, -0.245, DRAWER_FLOOR_Z + 0.020), (0.11, -0.10)),
)

OBJ_RGBA = {
    "plate":  (0.95, 0.95, 0.92, 1.0),
    "mug":    (0.88, 0.45, 0.25, 1.0),
    "bottle": (0.18, 0.55, 0.30, 1.0),
    "spoon":  (0.80, 0.82, 0.85, 1.0),
    "fork":   (0.85, 0.87, 0.90, 1.0),
}


@dataclass
class RandomizationConfig:
    pos_jitter: float = 0.03
    yaw_jitter: float = 0.40
    friction_scale: tuple[float, float] = (0.7, 1.3)
    mass_scale: tuple[float, float] = (0.75, 1.3)
    size_scale: tuple[float, float] = (0.92, 1.08)
    color_jitter: float = 0.05
    bg_hue_jitter: bool = True


def _yaw_quat(yaw: float) -> list[float]:
    """Quaternion for a rotation about world +z (w, x, y, z order)."""
    c = math.cos(yaw / 2.0)
    s = math.sin(yaw / 2.0)
    return [c, 0.0, 0.0, s]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class DinnerTableEnv:
    """Gym-style bimanual SO-101 dinner-table environment.

    The model is compiled from MjSpec on construction (and per
    reset-with-new-seed) so domain randomization changes the physics —
    object placement, mass, friction, size, colours — not just qpos.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    ARM_HOME = {
        # neutral folded pose: out of the way, facing the workspace
        "A": (0.0, -1.2, 1.4, -1.2, 0.0, 0.5),
        "B": (0.0, -1.2, 1.4, -1.2, 0.0, 0.5),
    }

    def __init__(self, seed: int = 0, randomize: bool = True,
                 randomization: RandomizationConfig | None = None,
                 render_mode: str | None = None,
                 max_episode_steps: int = 1500):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.randomize = randomize
        self.rand_cfg = randomization or RandomizationConfig()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.model, self.object_specs = self._build_scene()
        self.data = mujoco.MjData(self.model)
        self._resolve_ids()
        self._renderer = None
        self._poured = False
        self._elapsed = 0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #
    def _build_scene(self):
        spec = mujoco.MjSpec.from_file(WORLD_XML)

        # arms
        for arm in ARMS:
            arm_spec = mujoco.MjSpec.from_file(SO101_XML)
            arm_spec.meshdir = "."
            # disable collisions on the wrist camera mount boxes: they
            # are mount plates, not functional collision geometry, and
            # they block legitimate low grasps near the countertop edge
            # (found by the drawer-handle reach study). Visual meshes
            # still render the camera.
            for gname in ("camera_box1", "camera_box2"):
                g = arm_spec.geom(f"{gname}")
                g.contype = 0
                g.conaffinity = 0
            base = ARM_BASES[arm]
            frame = spec.worldbody.add_frame(
                pos=list(base["pos"]), quat=_yaw_quat(base["yaw"]))
            spec.attach(arm_spec, prefix=f"{arm}/", frame=frame)

        # task objects
        object_specs = []
        for name, start, goal in DEFAULT_OBJECTS:
            object_specs.append(self._add_object(spec, name, start, goal))

        # drawer actuator: a motor (force) actuator on the slide joint; the
        # skill layer gates it on gripper-handle contact for physical honesty
        spec.add_actuator(
            name="drawer_act",
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[1.0] + [0.0] * 9,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target="drawer_slide",
        )

        model = spec.compile()
        return model, object_specs

    def _rand(self, lo: float, hi: float, size=None):
        if not self.randomize:
            return np.full(size if size is not None else 1,
                           0.5 * (lo + hi)).squeeze()
        return self.rng.uniform(lo, hi, size)

    def _add_object(self, spec, name: str, start, goal):
        """Add one randomized task object; returns its spec dict.

        Every task object is a free body so grasping can actually move it.
        Utensils start inside the drawer and ride with it through ordinary
        contact/friction; welding them to the drawer would make a successful
        grasp physically impossible.
        """
        cfg = self.rand_cfg
        # _rand returns np.float64 (a Python-float subclass) — MuJoCo accepts them
        jx = self._rand(-cfg.pos_jitter, cfg.pos_jitter)
        jy = self._rand(-cfg.pos_jitter, cfg.pos_jitter)
        scale = self._rand(*cfg.size_scale)
        mass_scale = self._rand(*cfg.mass_scale)
        friction = self._rand(*cfg.friction_scale)
        if name in ("spoon", "fork"):
            # utensil pinches rely on grip friction: boost it
            friction = max(friction, self._rand(1.5, 2.2))
        yaw = self._rand(-cfg.yaw_jitter, cfg.yaw_jitter)

        if name in ("spoon", "fork"):
            # World coordinates inside the closed drawer.  Keep the long
            # utensil inside the tray bounds while allowing small x jitter.
            # Utensils are laid ACROSS the tray (yaw ±90°) at the -y end,
            # overhanging the -y wall so the arm can grasp the tail
            # horizontally from outside the tray.
            world_start = (
                float(np.clip(start[0] + 0.5 * jx, -0.590, -0.570)),
                start[1],
                start[2],
            )
            # yaw along the tray's x-axis so the tail overhangs the open
            # front (grasp corridor); jittered
            across = 0.5 * self._rand(-cfg.yaw_jitter, cfg.yaw_jitter)
            b = spec.worldbody.add_body(name=name, pos=list(world_start),
                                quat=_yaw_quat(across))
        else:
            world_start = (start[0] + jx, start[1] + jy, start[2])
            b = spec.worldbody.add_body(
                name=name, pos=list(world_start), quat=_yaw_quat(yaw)
            )

        b.add_freejoint()

        rgba = np.array(OBJ_RGBA[name])
        if self.randomize:
            rgba[:3] = np.clip(
                rgba[:3] + self._rand(-cfg.color_jitter, cfg.color_jitter, 3),
                0.0, 1.0)

        def add_geom(gname, gtype, size, pos=(0.0, 0.0, 0.0), rgba=None,
                     quat=None):
            if not hasattr(size, "__iter__"):
                size = (size, size, size)   # sphere shorthand
            kwargs = {}
            if quat is not None:
                kwargs["quat"] = list(quat)
            return b.add_geom(name=gname, type=gtype, size=list(size),
                              pos=list(pos),
                              rgba=list(rgba if rgba is not None else self._obj_rgba[name]),
                              contype=1, conaffinity=1, condim=3,
                              friction=[friction, 0.005, 0.0001], **kwargs)

        if name == "plate":
            r = 0.105 * scale
            # mjGEOM_CYLINDER size = (radius, half-height, unused)
            add_geom(name, mujoco.mjtGeom.mjGEOM_CYLINDER, (r, 0.008, 0.0))
            add_geom(f"{name}_rim", mujoco.mjtGeom.mjGEOM_CYLINDER,
                     (r, 0.004, 0.0), pos=(0.0, 0.0, 0.008))
            b.mass = 0.45 * mass_scale
        elif name == "mug":
            r = 0.033 * scale
            h = 0.055 * scale
            add_geom(name, mujoco.mjtGeom.mjGEOM_CYLINDER, (r, h / 2, 0.0))
            add_geom(f"{name}_handle", mujoco.mjtGeom.mjGEOM_BOX,
                     (0.007, 0.016, 0.012), pos=(0.0, -(r + 0.010), 0.0))
            b.mass = 0.20 * mass_scale
        elif name == "bottle":
            r = 0.030 * scale
            h = 0.145 * scale
            add_geom(name, mujoco.mjtGeom.mjGEOM_CYLINDER, (r, h / 2, 0.0))
            add_geom(f"{name}_neck", mujoco.mjtGeom.mjGEOM_CYLINDER,
                     (0.013, 0.014, 0.0), pos=(0.0, 0.0, h + 0.014))
            b.mass = 0.60 * mass_scale
        else:  # spoon / fork
            L = 0.155 * scale
            # capsule axis is local +z; rotate +90° about +y so it lies
            # along local +x (matches the head offset below)
            add_geom(name, mujoco.mjtGeom.mjGEOM_CAPSULE,
                     (0.006, L / 2 - 0.02), pos=(0.0, 0.0, 0.0),
                     quat=[0.70710678, 0.0, 0.70710678, 0.0])
            if name == "spoon":
                add_geom(f"{name}_head", mujoco.mjtGeom.mjGEOM_SPHERE,
                         (0.011, 0.011, 0.011), pos=(L / 2 - 0.02, 0.0, 0.0))
            else:
                add_geom(f"{name}_head", mujoco.mjtGeom.mjGEOM_BOX,
                         (0.018, 0.012, 0.003), pos=(L / 2 - 0.02, 0.0, 0.0))
            b.mass = 0.045 * mass_scale

        return {"name": name, "start": world_start, "goal": goal,
                "scale": scale, "mass_scale": mass_scale,
                "friction": friction, "yaw": yaw, "rgba": rgba}

    _obj_rgba = {k: list(v) for k, v in OBJ_RGBA.items()}

    # ------------------------------------------------------------------ #
    # ID resolution
    # ------------------------------------------------------------------ #
    def _resolve_ids(self):
        m = self.model
        self._arm_joint_ids: dict = {}
        self._arm_act_ids: dict = {}
        self._arm_site_ids: dict = {}
        self._obj_body_ids: dict = {}
        for arm in ARMS:
            self._arm_joint_ids[arm] = [mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}") for j in JOINTS]
            self._arm_act_ids[arm] = [mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}/{j}") for j in JOINTS]
            self._arm_site_ids[arm] = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/gripperframe")
        for o in self.object_specs:
            self._obj_body_ids[o["name"]] = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, o["name"])
        self._drawer_jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self._drawer_qadr = m.jnt_qposadr[self._drawer_jid]
        self._drawer_act_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_act")

    # ------------------------------------------------------------------ #
    # Reset / observation
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        """Reset. seed != None recompiles the scene with that randomization."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
            self.model, self.object_specs = self._build_scene()
            self.data = mujoco.MjData(self.model)
            self._resolve_ids()
        mujoco.mj_resetData(self.model, self.data)
        self._elapsed = 0
        self._poured = False
        self._drawer_latched_open = False
        for arm in ARMS:
            for jid, q in zip(self._arm_joint_ids[arm],
                              self.ARM_HOME[arm], strict=True):
                self.data.qpos[self.model.jnt_qposadr[jid]] = q
        self.data.qpos[self._drawer_qadr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        m, d = self.model, self.data
        obs = {}
        for arm in ARMS:
            obs[f"{arm}_qpos"] = np.array([
                d.qpos[m.jnt_qposadr[jid]] for jid in self._arm_joint_ids[arm]])
        obs["drawer"] = d.qpos[self._drawer_qadr]
        for o in self.object_specs:
            bid = self._obj_body_ids[o["name"]]
            obs[f"obj_{o['name']}"] = d.xpos[bid].copy()
        obs["task_state"] = self.task_state()
        return obs

    def gripper_site_pos(self, arm: str) -> np.ndarray:
        return self.data.site_xpos[self._arm_site_ids[arm]].copy()

    def set_pose_provider(self, provider, allow_fallback: bool = True) -> None:
        """Route object pose queries through an estimator.

        ``provider(name) -> (pos(3,), yaw | None) | None``. When set,
        ``object_pos`` / ``object_xaxis`` return the provider estimate.
        Development runs may opt into simulator fallback; strict evaluation
        raises when an estimate is unavailable. The skill layer consumes poses exclusively via
        these accessors, so swapping privileged state for perception is a
        one-line change at the orchestrator level.
        """
        self._pose_provider = provider
        self._pose_provider_allow_fallback = bool(allow_fallback)

    def _provider_pose(self, name: str):
        provider = getattr(self, "_pose_provider", None)
        if provider is None:
            return None
        try:
            est = provider(name)
        except KeyError:
            return None
        return est

    def object_pos(self, name: str) -> np.ndarray:
        est = self._provider_pose(name)
        if est is not None and est[0] is not None:
            return np.asarray(est[0], dtype=float).copy()
        if getattr(self, "_pose_provider", None) is not None and not getattr(
            self, "_pose_provider_allow_fallback", True
        ):
            raise LookupError(f"No camera-based pose available for {name!r}")
        return self.data.xpos[self._obj_body_ids[name]].copy()

    def object_yaw(self, name: str) -> float | None:
        est = self._provider_pose(name)
        if est is not None:
            return est[1]
        if getattr(self, "_pose_provider", None) is not None and not getattr(
            self, "_pose_provider_allow_fallback", True
        ):
            raise LookupError(f"No camera-based yaw available for {name!r}")
        m = self.data.xmat[self._obj_body_ids[name]].reshape(3, 3)
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def object_xaxis(self, name: str) -> np.ndarray:
        """Object body +x axis in world (utensil shaft direction)."""
        # A camera pose provider currently supplies yaw only, so preserve its
        # non-privileged horizontal estimate.  Without a provider, retain the
        # full simulator axis including roll/pitch; free utensils can tilt in
        # the tray and their head height then differs materially from the body
        # centre.
        if getattr(self, "_pose_provider", None) is not None:
            yaw = self.object_yaw(name)
            if yaw is not None:
                return np.array([math.cos(yaw), math.sin(yaw), 0.0])
        m = self.data.xmat[self._obj_body_ids[name]].reshape(3, 3)
        return m[:, 0].copy()

    # ------------------------------------------------------------------ #
    # Task-state machine
    # ------------------------------------------------------------------ #
    def task_state(self) -> dict:
        st: dict = {}
        st["drawer_open"] = bool(self.data.qpos[self._drawer_qadr] > 0.24)
        for o in self.object_specs:
            name = o["name"]
            # Evaluation is allowed to inspect simulator ground truth; policy
            # and skills still go through object_pos and its strict camera gate.
            pos = self.data.xpos[self._obj_body_ids[name]].copy()
            st[f"{name}_pos"] = pos
            if o["goal"] is not None:
                dist_xy = np.hypot(pos[0] - o["goal"][0], pos[1] - o["goal"][1])
                on_table = abs(pos[2] - COUNTERTOP_Z) < 0.08
                st[f"{name}_placed"] = bool(dist_xy < 0.055 and on_table)
            else:
                st[f"{name}_placed"] = None
        # pouring: bottle above the mug, tilted > 60 deg from vertical
        mug_p = st["mug_pos"]
        bot_p = st["bottle_pos"]
        bot_z = self.data.xmat[self._obj_body_ids["bottle"]].reshape(3, 3)[:, 2]
        above_mug = (np.hypot(bot_p[0] - mug_p[0], bot_p[1] - mug_p[1]) < 0.05
                     and bot_p[2] > mug_p[2])
        tilted = bot_z[2] < 0.5
        if above_mug and tilted:
            self._poured = True
        st["mug_filled"] = bool(self._poured)
        return st

    # ------------------------------------------------------------------ #
    # Stepping / control
    # ------------------------------------------------------------------ #
    def set_control(self, action) -> None:
        """13-dim control: [A(6), B(6), drawer_force].

        ``drawer_force`` is a generalized force (N) on the slide joint; the
        skill layer only commands it while the arm's gripper is in contact
        with the drawer handle (see policy/skills.py).
        """
        action = np.asarray(action, dtype=np.float64).reshape(13)
        for i, arm in enumerate(ARMS):
            for k, aid in enumerate(self._arm_act_ids[arm]):
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = np.clip(action[i * 6 + k], lo, hi)
        self.data.ctrl[self._drawer_act_id] = np.clip(action[12], -60.0, 60.0)

    def step(self, action=None):
        if action is not None:
            self.set_control(action)
        for _ in range(SIMSTEPS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)
            # Model the drawer's real-world open detent.  Once pulled past
            # the task threshold it cannot be shoved closed accidentally by
            # a gripper entering the tray, but remains a dynamic slide while
            # it is being opened.
            if self.data.qpos[self._drawer_qadr] >= 0.24:
                self._drawer_latched_open = True
            if self._drawer_latched_open \
                    and self.data.qpos[self._drawer_qadr] < 0.24:
                self.data.qpos[self._drawer_qadr] = 0.24
                dof = self.model.jnt_dofadr[self._drawer_jid]
                self.data.qvel[dof] = max(0.0, self.data.qvel[dof])
                mujoco.mj_forward(self.model, self.data)
        self._elapsed += 1
        st = self.task_state()
        return self._get_obs(), self._reward(st), self._done(), {"state": st}

    def gripper_handles_drawer(self, arm: str = "A") -> bool:
        """True when the arm's gripper geoms touch the drawer handle.

        Used to gate the drawer actuator so the drawer only moves while
        physically 'held' — keeping the manipulation honest.
        """
        m, d = self.model, self.data
        grip_prefix = {"A": "A/", "B": "B/"}[arm]
        for i in range(d.ncon):
            c = d.contact[i]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            names = {g1, g2}
            if "drawer_handle" in names and any(
                    (n or "").startswith(grip_prefix) for n in names):
                return True
        return False

    def _reward(self, st) -> float:
        r = 0.0
        if st["drawer_open"]:
            r += 0.5
        for name in ("plate", "mug", "spoon", "fork"):
            if st.get(f"{name}_placed"):
                r += 1.0
        if st["mug_filled"]:
            r += 1.0
        return r

    def _done(self) -> bool:
        return bool(self.success() or self._elapsed >= self.max_episode_steps)

    def success(self) -> bool:
        st = self.task_state()
        return bool(st["plate_placed"] and st["mug_placed"]
                    and st["spoon_placed"] and st["fork_placed"]
                    and st["mug_filled"])

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, camera: str = "overview", height: int = 720,
               width: int = 1280) -> np.ndarray:
        """Render the current scene from a named camera."""
        if (self._renderer is None
                or getattr(self, "_renderer_h", None) != height
                or getattr(self, "_renderer_w", None) != width):
            self._renderer = mujoco.Renderer(self.model, height=height,
                                             width=width)
            self._renderer_h, self._renderer_w = height, width
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def camera_obs(self, names=("overhead", "drawer_cam", "placemat_cam")):
        return {cam: self.render(camera=cam, height=224, width=224)
                for cam in names}

    def close(self) -> None:
        """Release the MuJoCo renderer deterministically."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(seed: int = 0, randomize: bool = True, **kwargs):
    return DinnerTableEnv(seed=seed, randomize=randomize, **kwargs)
