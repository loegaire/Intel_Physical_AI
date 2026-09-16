"""Manipulation skills for the bimanual dinner-table task.

Each skill is a small state machine producing 13-dim control actions
(12 arm joints + drawer force) at 50 Hz. The orchestrator
(``policy/orchestrator.py``) sequences skills from a natural-language
plan, closing the loop through the perception module.

Design notes
------------
* Every arm motion solves endpoint IK (with randomized restarts) and
  then commands a *joint-space rate-limited* trajectory toward it. A
  straight Cartesian descent can pass through unreachable mid-heights;
  a joint-interpolated path to a verified reachable endpoint cannot.
* Grasp/Place re-query the object pose every step through the
  perception module (swap point for the VLA policy).
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportMissingImports=false
# (mujoco's C extension ships incomplete stubs — mj_id2name, mjtObj, etc.
#  exist at runtime; intra-repo imports resolve via the package layout.)

from __future__ import annotations

import enum

import mujoco
import numpy as np
from envs.dinner_table_env import ARM_BASES, COUNTERTOP_Z, DinnerTableEnv

from policy.ik import solve_ik, ArmIK

# gripper joint control values
GRIP_OPEN = 1.20       # gripper qpos target when open
GRIP_CLOSED = -0.15    # near-fully closed

# per-object grasp offsets (local xy of the grasp point from body centre)
# plate: grasp the rim edge (offset toward the arm); mug: the handle side;
# spoon/fork: mid-shaft; bottle: mid-body.
GRASP_OFFSETS = {
    "plate": 0.097,     # rim ring radius (jaws straddle the rim edge)
    "mug": 0.036,        # near the handle / wall
    "bottle": 0.0,
    "spoon": 0.0,
    "fork": 0.0,
}

# Per-object grasp strategy. Offsets are radial from the object centre;
# None means the grasp pose is computed from body geometry (see
# Grasp._grasp_pose). Heights are where the jaws close, above centre.
GRASP_HEIGHTS = {
    "plate": 0.008,    # rim center (body half-height 0.008)
    "mug": 0.028,      # wall top (mug half-height 0.0275)
    "bottle": 0.0,     # mid-body
    "spoon": 0.0,      # head sphere centre (same plane as shaft)
    "fork": 0.0,       # shaft centre
}

CLEARANCE = 0.10       # m above countertop for safe transit
JOINT_RATE = 0.05      # rad per 50 Hz control step (2.5 rad/s)
OPEN_DRAWER_RATE = 0.10  # faster rate for drawer operations


class SkillStatus(enum.Enum):
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2


class Skill:
    """Base class: produce a 13-dim action; advance internal state."""

    def __init__(self, env: DinnerTableEnv, max_steps: int = 2000):
        self.env = env
        self.t = 0
        self.max_steps = max_steps
        self.status = SkillStatus.RUNNING

    def act(self) -> np.ndarray:
        self.t += 1
        action = self._act()
        if self.t > self.max_steps:            # hard skill timeout
            self.status = SkillStatus.FAILURE
        return action

    def _act(self) -> np.ndarray:              # pragma: no cover - abstract
        raise NotImplementedError

    # ---- shared helpers ---- #
    def qpos_of(self, arm: str) -> np.ndarray:
        """Current joint positions (6,) of one arm."""
        qadr = [self.env.model.jnt_qposadr[j]
                for j in self.env._arm_joint_ids[arm]]
        return np.array([self.env.data.qpos[a] for a in qadr])

    def ctrl_of(self, arm: str) -> np.ndarray:
        """Last commanded joint targets (6,) of one arm."""
        return np.array([self.env.data.ctrl[a]
                         for a in self.env._arm_act_ids[arm]])

    def _pack(self, qa, qb, drawer_force: float = 0.0) -> np.ndarray:
        return np.concatenate([qa, qb, [drawer_force]]).astype(np.float64)

    def _rate_limited(self, q_from: np.ndarray, q_to: np.ndarray,
                      rate: float = JOINT_RATE) -> np.ndarray:
        """Step q_from toward q_to by at most `rate` per joint."""
        delta = np.clip(q_to - q_from, -rate, rate)
        return q_from + delta

    def _two_arm_hold(self):
        """Current qpos of both arms (hold-everything action basis)."""
        return self.qpos_of("A"), self.qpos_of("B")

    def _reach_endpoint(self, arm: str, target, grip: float,
                        speed_hint: float = 0.0) -> np.ndarray:
        """Common motion pattern: IK to `target`, rate-limit toward it.

        Returns the 6-dim joint command for `arm` (gripper -> `grip`).
        Sets self.status FAILURE if endpoint IK fails.
        """
        res = solve_ik(self.env, arm, np.asarray(target, dtype=float),
                       q_init=self.qpos_of(arm)[:5],
                       restarts=12, steps=300, tol=0.006)
        if not res.ok:
            # no converged solution: hold position (do not chase a
            # best-of-bad random restart); fail only if far off
            if res.err > 0.03:
                self.status = SkillStatus.FAILURE
            return self.qpos_of(arm)
        q_cmd = self.qpos_of(arm).copy()
        q_cmd[:5] = self._rate_limited(q_cmd[:5], res.qpos)
        q_cmd[5] = grip
        return q_cmd


class MoveTo(Skill):
    """Move an arm's gripper site to a world-frame xyz target.

    Transit path: lift to clearance -> translate -> descend, each phase
    using endpoint IK + joint rate limiting.
    """

    def __init__(self, env, arm: str, target, grip: float = GRIP_OPEN,
                 max_steps: int = 700, tol: float = 0.015):
        super().__init__(env, max_steps)
        self.arm = arm
        self.target = np.asarray(target, dtype=float)
        self.grip = grip
        self.tol = tol
        gpos = env.gripper_site_pos(arm)
        self._clear_z = max(COUNTERTOP_Z + CLEARANCE,
                            self.target[2] + 0.02)
        self._start_xy = np.array([gpos[0], gpos[1]])
        self.phase = "lift" if gpos[2] < self._clear_z - 0.01 else "translate"

    def _phase_target(self) -> np.ndarray:
        if self.phase == "lift":
            return np.array([self._start_xy[0], self._start_xy[1],
                             self._clear_z])
        if self.phase == "translate":
            return np.array([self.target[0], self.target[1],
                             self._clear_z])
        return self.target.copy()

    def _act(self) -> np.ndarray:
        env = self.env
        gpos = env.gripper_site_pos(self.arm)
        tgt = self._phase_target()

        if self.phase == "lift" and gpos[2] >= self._clear_z - 0.02:
            self.phase = "translate"
            tgt = self._phase_target()
        elif self.phase == "translate":
            if (np.hypot(*(tgt[:2] - gpos[:2])) < 0.02
                    and abs(tgt[2] - gpos[2]) < 0.03):
                self.phase = "descend"
                tgt = self._phase_target()
        elif self.phase == "descend":
            if np.linalg.norm(self.target - gpos) < self.tol:
                self.status = SkillStatus.SUCCESS

        q_cmd = self._reach_endpoint(self.arm, tgt, self.grip)
        other = self.qpos_of("B" if self.arm == "A" else "A")
        return self._pack(q_cmd, other) if self.arm == "A" \
            else self._pack(other, q_cmd)


class Grasp(Skill):
    """Side grasp: point the gripper horizontally at the object, insert
    the jaws across it, close, and lift.

    Per-object grasp targets (all from live perception poses):

    * spoon / fork — the shaft (capsule), approached perpendicular to the
      shaft axis, jaws straddling it vertically;
    * plate — the rim ring, approached from outside the rim, jaws
      straddling the ring;
    * mug — the wall cylinder, approached radially, jaws across the wall;
    * bottle — mid-body, approached radially.

    The approach direction is the horizontal direction from the arm base
    toward the object, so the gripper naturally faces into the workspace.
    """

    APPROACH_STANDOFF = 0.10   # hover distance from the grasp point (m)

    def __init__(self, env, arm: str, obj_name: str,
                 lift_h: float = 0.12, max_steps: int = 1000):
        super().__init__(env, max_steps)
        self.arm = arm
        self.obj = obj_name
        self.lift_h = lift_h
        self.phase = "hover"
        self._phase_t = 0
        self._lift_base_z: float | None = None
        self._roll: float | None = None        # locked after alignment
        self._approach: np.ndarray | None = None
        self._grip_q: np.ndarray | None = None # qpos when jaws start closing
        # IK is deterministic for a phase target but comparatively expensive.
        # Keep one joint-space branch until the phase changes or perception
        # reports a material target displacement.
        self._ik_cache_phase: str | None = None
        self._ik_cache_target: np.ndarray | None = None
        self._ik_cache_q: np.ndarray | None = None
        self._lift_xy: np.ndarray | None = None

    # ---- geometry ---- #
    def _obj_spec(self) -> dict:
        for s in self.env.object_specs:
            if s["name"] == self.obj:
                return s
        raise KeyError(self.obj)

    def _body_x_axis(self) -> np.ndarray:
        return self.env.object_xaxis(self.obj)

    def _grasp_data(self) -> tuple:
        """Returns (grasp_point, approach_dir, desired_jaw_axis).

        grasp_point — where the jaws close; approach_dir — gripper z
        direction (unit); desired_jaw_axis — world direction the jaw
        opening should straddle (unit, xy) or None for any.
        """
        env = self.env
        op = env.object_pos(self.obj)
        spec = self._obj_spec()
        base_xy = np.array(ARM_BASES[self.arm]["pos"][:2])
        d = base_xy - op[:2]
        n = np.linalg.norm(d)
        to_base = d / n if n > 1e-6 else np.array([1.0, 0.0])
        approach_horizontal = np.array([-to_base[0], -to_base[1], 0.0])  # 3-D

        if self.obj == "plate":
            # Top-down grasp on the rim: approach vertically, jaws straddle
            # the rim ring. Grasp point is on the rim, offset from center
            # toward the arm base so jaws close on the rim edge.
            r = 0.105 * spec["scale"]
            gp = op + np.array([to_base[0] * r, to_base[1] * r, GRASP_HEIGHTS["plate"]])
            # grasp at rim, approach vertically
            approach = np.array([0.0, 0.0, -1.0])
            # jaw axis tangent to rim (horizontal)
            tangent = np.array([-to_base[1], to_base[0]])
            return gp, approach, tangent
        if self.obj == "mug":
            # Top-down grasp on the mug wall
            gp = op + np.array([0.0, 0.0, GRASP_HEIGHTS["mug"]])
            approach = np.array([0.0, 0.0, -1.0])
            # jaw axis radial (horizontal)
            return gp, approach, None
        if self.obj == "bottle":
            # Side grasp at mid-height
            gp = op.copy()
            approach = approach_horizontal
            return gp, approach, None
        if self.obj in ("spoon", "fork"):
            # Tray utensils lie ACROSS the tray, overhanging the -y wall.
            # Grasp the overhanging tail horizontally from -y: the grasp
            # point is on the shaft ~2 cm outside the wall (world y of
            # the wall ≈ -0.265).
            shaft3 = self._body_x_axis()
            ns = np.linalg.norm(shaft3)
            shaft3 = shaft3 / ns if ns > 1e-6 \
                else np.array([1.0, 0.0, 0.0])
            # utensils lie along the tray's x-axis with the tail
            # overhanging the open drawer front (+x): approach the +x
            # tail from above-front (26° down), jaws straddling the
            # shaft along world y (verified reachable by the workspace
            # feasibility study)
            # Grasp the +x shaft segment.  The head can tip down through the
            # slotted drawer front; staying 2 cm from the body centre keeps
            # the target on the supported, reachable part of the utensil.
            t = 0.020 * spec["scale"]
            p_a = op + shaft3 * t
            p_b = op - shaft3 * t
            # choose whichever end has larger x (the +x overhang)
            tail = p_a if p_a[0] > p_b[0] else p_b
            gp = tail.copy()
            # Top-down pinch: the raised tray floor is inside the arm's
            # reachable workspace and local +x is the insertion axis.
            approach = np.array([0.0, 0.0, -1.0])
            # jaw axis along world y: horizontal straddle of the shaft
            return gp, approach, np.array([0.0, 1.0])
        # fallback: radial approach at object height
        return op.copy(), approach_horizontal, None

    # ---- jaw alignment ---- #
    def _jaw_axis(self) -> np.ndarray:
        m, d = self.env.model, self.env.data
        g0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                               f"{self.arm}/fixed_jaw_box3")
        g1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                               f"{self.arm}/moving_jaw_sph_tip1")
        return (d.geom_xpos[g1] - d.geom_xpos[g0])[:2]

    def _jaw_axis_3d(self) -> np.ndarray:
        """Full 3-D jaw opening axis (used to align vertical pinches)."""
        m, d = self.env.model, self.env.data
        g0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                               f"{self.arm}/fixed_jaw_box3")
        g1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                               f"{self.arm}/moving_jaw_sph_tip1")
        return d.geom_xpos[g1] - d.geom_xpos[g0]

    def _align_roll(self, grasp_gp: np.ndarray, approach: np.ndarray,
                    desired_jaw) -> bool:
        """Sweep wrist_roll for the best jaw alignment at the hover pose,
        keeping the approach direction via oriented IK.

        Prefers a roll close to the IK solution's roll to avoid large
        joint-space jumps.

        ``desired_jaw`` semantics:
          * ndarray(2) — preferred jaw-axis direction in world xy;
          * "vertical" — the jaw opening should be vertical (one jaw
            above the shaft, one below) — scored on the full 3-D axis;
          * None — any orientation (falls back to a y-preference).
        """
        from policy.ik import solve_ik_oriented
        env = self.env
        hover = grasp_gp - approach * self.APPROACH_STANDOFF
        ok, q, _ = solve_ik_oriented(env, self.arm, hover, approach,
                                     q_init=self.qpos_of(self.arm)[:5],
                                     restarts=10, steps=800, tilt_tol=0.55)
        if not ok:
            return False
        qadr5 = [env.model.jnt_qposadr[j]
                 for j in env._arm_joint_ids[self.arm][:5]]
        backup = env.data.qpos[qadr5].copy()
        ik_roll = q[4]
        best_roll, best_score = ik_roll, -1.0
        vertical = isinstance(desired_jaw, str) and desired_jaw == "vertical"
        # Sweep around the IK roll ± π (but within joint limits)
        # Also include the IK roll itself as a candidate
        roll_candidates = [ik_roll]
        # Add nearby rolls within ±1.5 rad
        for delta in np.linspace(-1.5, 1.5, 13):
            if abs(delta) > 1e-6:
                roll_candidates.append(ik_roll + delta)
        # Clamp to joint limits
        roll_jid = env._arm_joint_ids[self.arm][4]
        lo, hi = env.model.jnt_range[roll_jid]
        roll_candidates = [np.clip(r, lo, hi) for r in roll_candidates]
        # Deduplicate
        roll_candidates = sorted(set(round(r, 6) for r in roll_candidates))
        
        for roll in roll_candidates:
            env.data.qpos[qadr5] = q
            env.data.qpos[qadr5[4]] = roll
            mujoco.mj_forward(env.model, env.data)
            if vertical:
                ax3 = self._jaw_axis_3d()
                n = np.linalg.norm(ax3)
                if n < 1e-9:
                    continue
                score = abs(ax3[2]) / n       # jaw axis along world z
            else:
                des = desired_jaw if desired_jaw is not None \
                    else np.array([0.0, 1.0])
                axis = self._jaw_axis()
                n_ax = np.linalg.norm(axis)
                n_des = np.linalg.norm(des)
                if n_ax < 1e-9 or n_des < 1e-9:
                    continue
                score = abs(np.dot(axis, des)) / (n_ax * n_des)
            if score > best_score:
                best_score, best_roll = score, roll
        env.data.qpos[qadr5] = backup
        mujoco.mj_forward(env.model, env.data)
        self._roll = best_roll
        return True

    # ---- main loop ---- #
    def _act(self) -> np.ndarray:
        from policy.ik import solve_ik_fixed_roll, solve_ik_oriented
        env = self.env
        self._phase_t += 1
        motion_phase = self.phase
        gpos = env.gripper_site_pos(self.arm)
        gp, approach, desired_jaw = self._grasp_data()
        if self._approach is None:
            self._approach = np.asarray(approach, dtype=float).copy()
        approach = self._approach        # frozen after first step
        grip = GRIP_OPEN

        if self.phase == "hover":
            # move to the standoff point facing the object
            hover = gp - approach * self.APPROACH_STANDOFF
            if np.linalg.norm(hover - gpos) < 0.02 or self._phase_t > 200:
                if self._align_roll(gp, self._approach, desired_jaw):
                    self.phase, self._phase_t = "insert", 0
                else:
                    self.status = SkillStatus.FAILURE
            tgt = hover
        elif self.phase == "insert":
            # stop with the grasp point BETWEEN the open jaws.
            # For vertical approach (top-down): fixed jaw is ~7 mm below site
            # when open, so site should be ~7 mm above grasp point.
            # For horizontal approach: jaw tips extend ~5 cm forward from site.
            if self.obj in ("spoon", "fork"):
                # gripperframe is slightly offset from the closed fingertip
                # centre in the primitive SO-101 model.
                tgt = gp + np.array([-0.004, -0.008, -0.004])
            else:
                jaw_ext = 0.007 if abs(approach[2]) > 0.7 else 0.050
                tgt = gp - approach * jaw_ext
            grip = GRIP_OPEN
            if np.linalg.norm(tgt - gpos) < 0.015 or self._phase_t > 150:
                self.phase, self._phase_t = "close", 0
                self._grip_q = self.qpos_of(self.arm)
        elif self.phase == "close":
            # hold position once contact is made — keep pressing and
            # the reaction shoves the object away
            tgt = gpos.copy()
            grip = GRIP_CLOSED
            # Contact itself is the meaningful closure criterion: a real
            # object stops the gripper well before its empty-jaw angle.
            if self._grip_contact() or self._phase_t > 120:
                self.phase, self._phase_t = "lift", 0
                self._lift_base_z = env.object_pos(self.obj)[2]
                self._lift_xy = gpos[:2].copy()
        elif self.phase == "lift":
            base = self._lift_base_z if self._lift_base_z is not None \
                else env.object_pos(self.obj)[2]
            lift_xy = self._lift_xy if self._lift_xy is not None else gpos[:2]
            tgt = np.array([lift_xy[0], lift_xy[1], base + self.lift_h])
            grip = GRIP_CLOSED
            if np.linalg.norm(tgt - gpos) < 0.03 or self._phase_t > 250:
                obj_now = env.object_pos(self.obj)
                if obj_now[2] > base + 0.04:
                    self.status = SkillStatus.SUCCESS
                else:
                    self.status = SkillStatus.FAILURE
        else:                                   # pragma: no cover
            tgt = gpos.copy()

        # Closing only changes the gripper joint; holding the current arm pose
        # avoids both needless IK work and tiny Cartesian servo oscillations.
        if motion_phase == "close":
            q_cmd = self.qpos_of(self.arm).copy()
            q_cmd[5] = grip
        else:
            cache_valid = (
                self._ik_cache_phase == motion_phase
                and self._ik_cache_target is not None
                and self._ik_cache_q is not None
                and np.linalg.norm(tgt - self._ik_cache_target) < 0.015
            )
            if cache_valid:
                q5 = self._ik_cache_q
            else:
                q5 = None

        # Oriented IK to the phase target, roll pinned after alignment.  Solve
        # only on a cache miss, then rate-limit toward that stable solution.
        if motion_phase != "close" and q5 is None and self._roll is not None \
                and motion_phase in ("insert", "lift"):
            res = solve_ik_fixed_roll(
                env, self.arm, tgt, self._roll,
                q_init=self.qpos_of(self.arm)[:4], restarts=4, steps=200)
            if res.ok:
                q5 = res.qpos.copy()
            else:
                self.status = SkillStatus.FAILURE
        elif motion_phase != "close" and q5 is None:
            approach_use = self._approach if self._approach is not None \
                else approach
            tilt_tol = 0.55 if motion_phase in ("hover", "insert") else 0.30
            # hover phase needs more exploration to escape local minima
            restarts = 12 if motion_phase == "hover" else 6
            steps = 800 if motion_phase == "hover" else 600
            ok, q5, _ = solve_ik_oriented(
                env, self.arm, tgt, approach_use,
                q_init=self.qpos_of(self.arm)[:5], restarts=restarts,
                tilt_tol=tilt_tol, steps=steps)
            if not ok:
                self.status = SkillStatus.FAILURE

        if motion_phase != "close":
            if q5 is not None:
                if not cache_valid:
                    self._ik_cache_phase = motion_phase
                    self._ik_cache_target = tgt.copy()
                    self._ik_cache_q = q5.copy()
                q_cmd = self.qpos_of(self.arm).copy()
                q_cmd[:5] = self._rate_limited(q_cmd[:5], q5)
                q_cmd[5] = grip
            else:
                q_cmd = self.qpos_of(self.arm).copy()
                q_cmd[5] = grip

        other = self.qpos_of("B" if self.arm == "A" else "A")
        return self._pack(q_cmd, other) if self.arm == "A" \
            else self._pack(other, q_cmd)

    def _grip_contact(self) -> bool:
        """True when the arm's gripper geoms touch the target object."""
        m, d = self.env.model, self.env.data
        prefix = f"{self.arm}/"
        for i in range(d.ncon):
            c = d.contact[i]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            n1, n2 = g1 or "", g2 or ""
            if n1.startswith(prefix) or n2.startswith(prefix):
                other_name = n2 if n1.startswith(prefix) else n1
                if other_name == self.obj or \
                        other_name.startswith(f"{self.obj}_"):
                    return True
        return False


class Place(Skill):
    """Carrying an object: move above the goal, descend, open, retreat."""

    def __init__(self, env, arm: str, obj_name: str, goal_xy,
                 place_z: float | None = None, max_steps: int = 800,
                 goal_tol: float = 0.06):
        super().__init__(env, max_steps)
        self.arm = arm
        self.obj = obj_name
        self.goal_xy = np.asarray(goal_xy, dtype=float)
        self.place_z = place_z
        self.goal_tol = goal_tol
        self.phase = "approach"
        self._phase_t = 0

    def _act(self) -> np.ndarray:
        env = self.env
        self._phase_t += 1
        gpos = env.gripper_site_pos(self.arm)
        op = env.object_pos(self.obj)
        z = self.place_z if self.place_z is not None else op[2]
        grip = GRIP_CLOSED            # keep carrying

        if self.phase == "approach":
            tgt = np.array([self.goal_xy[0], self.goal_xy[1], z + 0.10])
            if np.linalg.norm(tgt - gpos) < 0.02:
                self.phase, self._phase_t = "descend", 0
        elif self.phase == "descend":
            tgt = np.array([self.goal_xy[0], self.goal_xy[1], z])
            if np.linalg.norm(tgt - gpos) < 0.015:
                self.phase, self._phase_t = "release", 0
        elif self.phase == "release":
            tgt = gpos.copy()
            grip = GRIP_OPEN
            if self._phase_t > 30:     # let the object settle open-handed
                self.phase, self._phase_t = "retreat", 0
        elif self.phase == "retreat":
            tgt = np.array([gpos[0], gpos[1], z + 0.12])
            grip = GRIP_OPEN
            if np.linalg.norm(tgt - gpos) < 0.03 or self._phase_t > 100:
                op_now = env.object_pos(self.obj)
                dist = np.hypot(op_now[0] - self.goal_xy[0],
                                op_now[1] - self.goal_xy[1])
                self.status = (SkillStatus.SUCCESS if dist < self.goal_tol
                               else SkillStatus.FAILURE)
        else:                                   # pragma: no cover
            tgt = gpos.copy()

        q_cmd = self._reach_endpoint(self.arm, tgt, grip)
        other = self.qpos_of("B" if self.arm == "A" else "A")
        return self._pack(q_cmd, other) if self.arm == "A" \
            else self._pack(other, q_cmd)


class OpenDrawer(Skill):
    """Side-grasp the vertical handle post and pull the tray open (+x).

    The handle is a vertical post on the drawer front's +x face.  Each
    Cartesian phase caches one position-IK branch so the joint trajectory
    remains smooth: transit -> descend -> insert -> close -> pull.  The arm
    leads the moving post while a small drawer motor compensates friction.
    """

    HANDLE_POS_CLOSED = np.array([-0.398, -0.162, 0.758])  # world frame

    def __init__(self, env, arm: str = "A", open_qpos: float = 0.26,
                 pull_force: float = 4.0, max_steps: int = 700):
        super().__init__(env, max_steps)
        self.arm = arm
        self.open_qpos = open_qpos
        self.pull_force = pull_force
        self.phase = "transit"
        self._phase_t = 0
        self._q_sol: np.ndarray | None = None   # cached IK for this phase
        self._q_tgt: np.ndarray | None = None    # target the cache solved
        self._held_once = False                  # verified handle contact
        self._roll: float | None = None          # locked pinch-plane roll
        # Transit subphase for phased motion
        self._transit_subphase = "lift"
        self._transit_q_target: np.ndarray | None = None

    def _handle_pos(self) -> np.ndarray:
        q = self.env.data.qpos[self.env._drawer_qadr]
        pos = self.HANDLE_POS_CLOSED.copy()
        pos[0] += q
        return pos

    def _set_phase(self, new_phase: str) -> None:
        self.phase = new_phase
        self._phase_t = 0
        self._q_sol = None            # force a fresh IK for the new target
        self._q_tgt = None
        # Pre-compute transit target joint configuration
        if new_phase == "transit":
            self._transit_q_target = None
            self._transit_subphase = "lift"
            if hasattr(self, '_lift_xy'):
                delattr(self, '_lift_xy')
            self._transit_subphase = "lift"

    def _act(self) -> np.ndarray:
        env = self.env
        self._phase_t += 1
        gpos = env.gripper_site_pos(self.arm)
        hp = self._handle_pos()
        grip = GRIP_OPEN

        # Transit route (avoids the tableware on the counter and the
        # countertop edge): high transit over the cabinet gap -> descend
        # in the gap between cabinet and countertop -> slide onto the
        # handle -> close -> pull.
        # Vertical-post side-grasp plan. Jaws extend ~5 cm past the site
        # along the gripper's -z; the post face is at hp[0]+0.0125, so the
        # close-phase site sits at POST_X with tips wrapping the post.
        PRE_X = -0.34          # pre-grasp column inside measured arm-A workspace
        # The post is offset toward the drawer's front edge so that the
        # gripper-site path at y=-0.18 places it between both jaws.
        PRE_Y = -0.18
        global PRE_X_LOCAL
        PRE_X_LOCAL = -0.34
        CLOSE_X = -0.40        # jaw tips extend toward the post at x=-0.45
        if self.phase == "transit":
            # Phased transit like MoveTo: lift -> translate -> descend
            if self._transit_subphase == "lift":
                if not hasattr(self, '_lift_xy'):
                    self._lift_xy = gpos[:2].copy()
                tgt = np.array([self._lift_xy[0], self._lift_xy[1], 0.95])  # lift to clearance
                if gpos[2] >= 0.93 or self._phase_t > 150:
                    self._transit_subphase = "translate"
                    self._transit_q_target = None  # recompute for translate
                    delattr(self, '_lift_xy')
                    tgt = np.array([PRE_X, PRE_Y, 0.95])
            elif self._transit_subphase == "translate":
                tgt = np.array([PRE_X, PRE_Y, 0.95])  # translate at clearance height
                if (np.hypot(tgt[0] - gpos[0], tgt[1] - gpos[1]) < 0.03
                        and abs(tgt[2] - gpos[2]) < 0.03) or self._phase_t > 300:
                    self._transit_subphase = "descend"
                    self._transit_q_target = None  # recompute for descend
                    tgt = np.array([PRE_X, PRE_Y, 0.85])
            else:  # descend
                tgt = np.array([PRE_X, PRE_Y, 0.85])  # descend to final transit height
                if np.linalg.norm(tgt - gpos) < 0.03 or self._phase_t > 450:
                    self._set_phase("descend")
                    tgt = np.array([PRE_X, PRE_Y, hp[2]])
        elif self.phase == "descend":
            # down the pre-grasp column to post-centre height
            tgt = np.array([PRE_X, PRE_Y, hp[2]])
            if np.linalg.norm(tgt - gpos) < 0.02 or self._phase_t > 250:
                self._align_roll_for_post(hp)
                self._set_phase("insert")
                tgt = np.array([CLOSE_X, PRE_Y, hp[2]])
        elif self.phase == "insert":
            # slide -x so the jaws wrap the post
            tgt = np.array([CLOSE_X, PRE_Y, hp[2]])
            if (env.gripper_handles_drawer(self.arm)
                    or self._phase_t > 150):
                self._set_phase("close")
        elif self.phase == "close":
            tgt = np.array([CLOSE_X, PRE_Y, hp[2]])
            grip = GRIP_CLOSED
            if env.gripper_handles_drawer(self.arm):
                self._held_once = True
            if (self._held_once and self._grip_closing_done()) \
                    or self._phase_t > 100:
                self._set_phase("pull")
        elif self.phase == "pull":
            # Pull ahead of the measured post rather than merely following
            # it.  The lead makes the arm servo provide the opening force;
            # the small drawer motor only compensates slide friction.
            pull_lead = 0.06
            tgt = np.array([CLOSE_X + (self._handle_pos()[0]
                                       - self.HANDLE_POS_CLOSED[0])
                            + pull_lead,
                            PRE_Y, hp[2]])
            grip = GRIP_CLOSED
            if env.data.qpos[env._drawer_qadr] >= self.open_qpos - 0.015:
                self.status = SkillStatus.SUCCESS
        else:                                   # pragma: no cover
            tgt = gpos.copy()

        # IK per phase; during pull the target rides ahead of the handle and
        # is therefore re-solved every step.
        # For transit lift: compute target joint config once, then rate-limit.
        # For transit translate/descend: use ArmIK for continuous control.
        from policy.ik import solve_ik
        if self.phase in ("pull",):
            self._q_sol = None
        q_sol = self._q_sol
        if self.phase == "transit":
            if self._transit_subphase == "lift":
                # Compute lift target joint config once
                if self._transit_q_target is None:
                    res = solve_ik(
                        self.env, self.arm, tgt,
                        q_init=self.qpos_of(self.arm)[:5],
                        restarts=12, steps=300, tol=0.006)
                    if not res.ok:
                        self.status = SkillStatus.FAILURE
                        return self._pack(*self._two_arm_hold())
                    self._transit_q_target = res.qpos.copy()
                # Rate-limit towards pre-computed target
                q_cmd = self.qpos_of(self.arm).copy()
                q_cmd[:5] = self._rate_limited(q_cmd[:5], self._transit_q_target, rate=OPEN_DRAWER_RATE)
                q_cmd[5] = grip
            else:
                # Keep one IK branch for the whole subphase. Re-solving from
                # scratch can alternate between valid joint basins, causing
                # the rate-limited shoulder command to oscillate around zero.
                if self._transit_q_target is None:
                    res = solve_ik(
                        self.env, self.arm, tgt,
                        q_init=self.qpos_of(self.arm)[:5],
                        restarts=12, steps=300, tol=0.006)
                    if not res.ok:
                        self.status = SkillStatus.FAILURE
                        return self._pack(*self._two_arm_hold())
                    self._transit_q_target = res.qpos.copy()
                q_cmd = self.qpos_of(self.arm).copy()
                q_cmd[:5] = self._rate_limited(
                    q_cmd[:5], self._transit_q_target,
                    rate=OPEN_DRAWER_RATE,
                )
                q_cmd[5] = grip
        elif q_sol is None:
            res = solve_ik(
                self.env, self.arm, tgt,
                q_init=self.qpos_of(self.arm)[:5],
                restarts=12, steps=300, tol=0.008,
            )
            ok = res.ok
            q5 = res.qpos if res.ok else None
            if not ok:
                # transient IK failure: hold and retry next step (the arm
                # pose drifts slightly as it settles, changing the warm
                # start basin); give up only after many consecutive misses
                self._ik_misses = getattr(self, "_ik_misses", 0) + 1
                if self._ik_misses > 30:
                    self.status = SkillStatus.FAILURE
                return self._pack(*self._two_arm_hold())
            self._ik_misses = 0
            if q5 is None:                     # defensive; unreachable
                return self._pack(*self._two_arm_hold())
            q_sol = q5
            self._q_sol = q_sol
        if self.phase != "transit":
            q_cmd = self.qpos_of(self.arm).copy()
            q_cmd[:5] = self._rate_limited(q_cmd[:5], q_sol, rate=OPEN_DRAWER_RATE)
            q_cmd[5] = grip
        def _holding() -> bool:
            # Power-assist drawer: the gripper verified contact in the
            # close phase; during the pull the actuator carries the load
            # while the gripper stays closed on the post. Pinch contact
            # flickers under servo load, so gate on proximity + closed
            # jaws (the arm visibly holds the post through the pull).
            if self.phase == "pull" and self._held_once:
                gp = env.gripper_site_pos(self.arm)
                hp_now = self._handle_pos()
                near = np.linalg.norm(gp - hp_now).item() < 0.08
                closed = self.qpos_of(self.arm)[5] < 0.2
                return bool(near and closed)
            if self.phase == "close":
                return env.gripper_handles_drawer(self.arm)
            return False

        if self.phase == "close" and env.gripper_handles_drawer(self.arm):
            self._held_once = True
        force = self.pull_force if (self.phase == "pull" and _holding()) \
            else 0.0
        other = self.qpos_of("B" if self.arm == "A" else "A")
        return self._pack(q_cmd, other, force) if self.arm == "A" \
            else self._pack(other, q_cmd, force)

    def _grip_closing_done(self) -> bool:
        """Gripper joint has closed near its target (jaws seated)."""
        return bool(self.qpos_of(self.arm)[5] < 0.2)

    def _align_roll_for_post(self, hp: np.ndarray) -> None:
        """Retain the continuous wrist branch reached during descent."""
        del hp
        self._roll = float(self.qpos_of(self.arm)[4])


class Pour(Skill):
    """Tilt the bottle over the mug until the task state flags a pour.

    The arm holding the bottle hovers its neck above the mug and rotates
    ``wrist_roll``; ``env.task_state()`` detects the pour geometry.
    """

    def __init__(self, env, arm: str, hold_arm: str = "B",
                 tilt_rate: float = 0.03, max_steps: int = 900):
        super().__init__(env, max_steps)
        self.arm = arm              # arm holding the bottle
        self.hold_arm = hold_arm
        self.tilt_rate = tilt_rate  # rad per control step
        self.phase = "over_mug"
        self._tilt = 0.0

    def _act(self) -> np.ndarray:
        env = self.env
        gpos = env.gripper_site_pos(self.arm)
        mug = env.object_pos("mug")
        neck_offset = np.array([0.0, 0.0, 0.17])

        if self.phase == "over_mug":
            # place the bottle NECK above the mug rim
            tgt = mug + neck_offset
            neck_now = gpos + neck_offset
            if np.linalg.norm(tgt - neck_now) < 0.03:
                self.phase = "tilt"
        elif self.phase == "tilt":
            if env.task_state()["mug_filled"]:
                self.status = SkillStatus.SUCCESS
        else:                                   # pragma: no cover
            pass

        tgt = (mug + neck_offset) if self.phase == "over_mug" else gpos
        q_cmd = self._reach_endpoint(self.arm, tgt, GRIP_CLOSED)

        if self.phase == "tilt":
            jid = self.env._arm_joint_ids[self.arm][4]
            lo, hi = self.env.model.jnt_range[jid]
            self._tilt += self.tilt_rate
            q_cmd[4] = np.clip(self.qpos_of(self.arm)[4] + self._tilt,
                               lo, hi)

        other = self.qpos_of(self.hold_arm)
        return self._pack(q_cmd, other) if self.arm == "A" \
            else self._pack(other, q_cmd)
