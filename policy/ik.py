"""IK solvers for the SO-101 arms in the dinner-table environment.

Provides:

* ``solve_ik`` — damped-least-squares position IK on an arm's
  ``gripperframe`` site, with joint-limit clamping.
* ``ArmIK`` — convenience wrapper caching per-arm IDs, with
  position+orientation-tolerance targeting used by the skill layer.

The solver is pure NumPy + MuJoCo Jacobians: cheap enough for 50 Hz
control on an Intel CPU, and structurally identical to what the
OpenVINO-optimized policy pipeline uses at inference time.
"""

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
# (mujoco's C extension ships incomplete stubs: mj_forward, mj_jacSite,
#  mj_name2id, mjtObj etc. all exist at runtime — verified by the test suite.)

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    ok: bool
    qpos: np.ndarray            # 5-DoF arm joints (no gripper)
    err: float                  # final position error norm (m)
    iters: int


def _norm_err(err: np.ndarray) -> float:
    """float() conversion is restricted by lint rules; use .item()."""
    return np.linalg.norm(err).item()


def solve_ik(env, arm: str, target: np.ndarray, q_init: np.ndarray | None = None,
             steps: int = 250, tol: float = 0.005,
             damping: float = 1e-4, pos_weight: float = 0.15,
             restarts: int = 8, seed: int | None = None) -> IKResult:
    """Damped-least-squares position IK on ``<arm>/gripperframe``.

    DLS IK is greedy and can stall at joint limits; on failure the solver
    retries from randomized starts (bounded by ``restarts``) and keeps the
    best solution. The world state is restored before returning.
    """
    model, data = env.model, env.data
    sid = env._arm_site_ids[arm]
    jids = env._arm_joint_ids[arm][:5]           # 5 DoF (gripper excluded)
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    backup = data.qpos[qadr].copy()
    target = np.asarray(target, dtype=float)

    rng = np.random.default_rng(seed if seed is not None else 0)
    starts: list = []
    if q_init is not None:
        starts.append(np.asarray(q_init, dtype=float))
    # randomized starts within joint limits
    for _ in range(max(0, restarts)):
        lo, hi = model.jnt_range[jids].T
        starts.append(rng.uniform(lo, hi))

    best: IKResult | None = None
    for start in starts:
        data.qpos[qadr] = start
        err_norm = np.inf
        iters_used = steps
        ok = False
        for it in range(steps):
            mujoco.mj_forward(model, data)
            err = target - data.site_xpos[sid]
            err_norm = _norm_err(err)
            if err_norm < tol:
                ok, iters_used = True, it
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, sid)
            J = jacp[:, qadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            alpha = min(1.0, pos_weight / max(1e-9, err_norm))
            for i, a in enumerate(qadr):
                lo, hi = model.jnt_range[jids[i]]
                data.qpos[a] = np.clip(data.qpos[a] + alpha * dq[i], lo, hi)
        q_out = data.qpos[qadr].copy()
        if best is None or (ok and not best.ok) or \
                (ok == best.ok and err_norm < best.err):
            best = IKResult(ok=ok, qpos=q_out, err=err_norm, iters=iters_used)
        if ok:
            break

    data.qpos[qadr] = backup
    mujoco.mj_forward(model, data)
    if best is None:                     # defensive: starts is never empty
        return IKResult(ok=False, qpos=backup.copy(), err=np.inf, iters=steps)
    return best


class ArmIK:
    """Per-arm IK helper: warm-starts from the arm's CURRENT pose so
    solutions stay continuous (no configuration jumps between steps).
    Falls back to randomized restarts only when warm start fails."""

    def __init__(self, env, arm: str):
        self.env = env
        self.arm = arm
        self._last_q: np.ndarray | None = None

    def solve(self, target: np.ndarray, **kwargs) -> IKResult:
        target = np.asarray(target, dtype=float)
        # 1) warm start from the live arm pose (continuity)
        q_cur = self.current_qpos()
        res = solve_ik(self.env, self.arm, target, q_init=q_cur,
                       restarts=0, **kwargs)
        if res.ok:
            self._last_q = res.qpos.copy()
            return res
        # 2) retry from the last successful solution
        if self._last_q is not None:
            res = solve_ik(self.env, self.arm, target, q_init=self._last_q,
                           restarts=0, **kwargs)
            if res.ok:
                self._last_q = res.qpos.copy()
                return res
        # 3) randomized restarts (rare; may jump configuration)
        res = solve_ik(self.env, self.arm, target, restarts=8, **kwargs)
        if res.ok:
            self._last_q = res.qpos.copy()
        return res

    def current_qpos(self) -> np.ndarray:
        ids = self.env._arm_joint_ids[self.arm][:5]
        qadr = [self.env.model.jnt_qposadr[j] for j in ids]
        return np.array([self.env.data.qpos[a] for a in qadr])

    def reset_cache(self):
        self._last_q = None


def solve_ik_fixed_roll(env, arm: str, target: np.ndarray, roll: float,
                        q_init: np.ndarray | None = None,
                        steps: int = 250, tol: float = 0.006,
                        damping: float = 1e-4, pos_weight: float = 0.15,
                        restarts: int = 8) -> IKResult:
    """Position IK with ``wrist_roll`` pinned to ``roll`` (4-DoF solve).

    Used for orientation-aligned grasps (e.g. plate rim: jaw axis tangent
    to the rim). Joint order: [pan, lift, elbow, flex] solved, roll fixed.
    """
    model, data = env.model, env.data
    sid = env._arm_site_ids[arm]
    jids = env._arm_joint_ids[arm][:4]           # 4 DoF, roll excluded
    roll_jid = env._arm_joint_ids[arm][4]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    roll_adr = model.jnt_qposadr[roll_jid]
    backup = data.qpos[np.concatenate([qadr, [roll_adr]])].copy()
    target = np.asarray(target, dtype=float)

    rng = np.random.default_rng(0)
    starts: list = []
    if q_init is not None:
        starts.append(np.asarray(q_init, dtype=float)[:4])
    lo4, hi4 = model.jnt_range[jids].T
    for _ in range(max(0, restarts)):
        starts.append(rng.uniform(lo4, hi4))

    best: IKResult | None = None
    for start in starts:
        data.qpos[qadr] = start
        data.qpos[roll_adr] = roll
        err_norm, iters_used, ok = np.inf, steps, False
        for it in range(steps):
            mujoco.mj_forward(model, data)
            err = target - data.site_xpos[sid]
            err_norm = _norm_err(err)
            if err_norm < tol:
                ok, iters_used = True, it
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, sid)
            J = jacp[:, qadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            alpha = min(1.0, pos_weight / max(1e-9, err_norm))
            for i, a in enumerate(qadr):
                lo, hi = model.jnt_range[jids[i]]
                data.qpos[a] = np.clip(data.qpos[a] + alpha * dq[i], lo, hi)
        q_out = np.append(data.qpos[qadr].copy(), roll)
        if best is None or (ok and not best.ok) or \
                (ok == best.ok and err_norm < best.err):
            best = IKResult(ok=ok, qpos=q_out, err=err_norm, iters=iters_used)
        if ok:
            break

    data.qpos[np.concatenate([qadr, [roll_adr]])] = backup
    mujoco.mj_forward(model, data)
    if best is None:
        return IKResult(ok=False, qpos=backup.copy(), err=np.inf, iters=steps)
    return best


def solve_ik_oriented(env, arm: str, target: np.ndarray, approach: np.ndarray,
                      steps: int = 600, tol: float = 0.010,
                      damping: float = 5e-3, qw: float = 0.5,
                      tilt_tol: float = 0.30, q_init: np.ndarray | None = None,
                      restarts: int = 10, seed: int = 0):
    """Position + approach-axis IK on ``<arm>/gripperframe`` (5-DoF).

    Solves for the site position AND rotates the gripper's z-axis (its
    approach direction) toward ``approach``. The orientation error is
    weighted by ``qw`` and only the first two cross-product components
    are used, giving a well-conditioned 5-D task for the 5 arm joints.

    Returns (ok, qpos_5, iters); world state is restored on return.
    """
    model, data = env.model, env.data
    sid = env._arm_site_ids[arm]
    jids = env._arm_joint_ids[arm][:5]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    backup = data.qpos[qadr].copy()
    target = np.asarray(target, dtype=float)
    approach = np.asarray(approach, dtype=float)
    approach = approach / np.linalg.norm(approach)

    rng = np.random.default_rng(seed)
    starts: list = []
    if q_init is not None:
        starts.append(np.asarray(q_init, dtype=float))
    lo5, hi5 = model.jnt_range[jids].T
    for _ in range(max(0, restarts)):
        starts.append(rng.uniform(lo5, hi5))

    for st in starts:
        data.qpos[qadr] = st
        for it in range(steps):
            mujoco.mj_forward(model, data)
            R = data.site_xmat[sid].reshape(3, 3)
            gz = R[2]
            perr = target - data.site_xpos[sid]
            oerr = np.cross(gz, approach)
            err = np.concatenate([perr, oerr[:2] * qw])
            if np.linalg.norm(perr) < tol and np.linalg.norm(oerr) < tilt_tol:
                q_out = data.qpos[qadr].copy()
                data.qpos[qadr] = backup
                mujoco.mj_forward(model, data)
                return True, q_out, it
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J5 = np.vstack([jacp[:, qadr], jacr[:2, qadr]])
            dq = J5.T @ np.linalg.solve(J5 @ J5.T + damping * np.eye(5), err)
            alpha = 0.25
            for i, a in enumerate(qadr):
                lo, hi = model.jnt_range[jids[i]]
                data.qpos[a] = np.clip(data.qpos[a] + alpha * dq[i], lo, hi)

    data.qpos[qadr] = backup
    mujoco.mj_forward(model, data)
    return False, backup.copy(), steps
