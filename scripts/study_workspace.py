"""Workspace feasibility study for the bimanual dinner-table scene.

Answers, numerically, the design questions behind `envs/world.xml` and
`envs/dinner_table_env.py`:

  * Where can the two SO-101 arm bases sit so both share the workspace?
  * Which arm should own which subtask (drawer / placemat / handoff)?
  * Are all object start & goal poses reachable with the 6-DoF arm?

Uses damped-least-squares IK on the `gripperframe` site with the exact
Menagerie SO-101 model that the environment attaches.

Run:  python scripts/study_workspace.py
"""

# mujoco's C-extension type stubs are incomplete: MjSpec, MjData, mj_name2id,
# mj_forward, mj_jacSite, mjtObj, Renderer etc. exist at runtime but are
# missing from the shipped typings. Suppress the false positives here.
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false

import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WORLD_XML = os.path.join(ROOT, "envs", "world.xml")
SO101_XML = os.path.join(ROOT, "assets", "so101", "so101.xml")

ARMS = ("A", "B")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll")

# Base placements under study (world frame; z = countertop height)
BASES = {
    "A": {"pos": (-0.22, -0.34, 0.75), "yaw": math.atan2(0.34, 0.22)},
    "B": {"pos": (0.10, 0.34, 0.75), "yaw": math.atan2(-0.34, -0.10)},
}

# Key workspace targets (world frame)
TARGETS = {
    "drawer_interior":  np.array([-0.46, 0.00, 0.71]),
    "drawer_front":     np.array([-0.31, 0.00, 0.68]),
    "handoff_centre":   np.array([-0.08, 0.00, 0.86]),
    "placemat_plate":   np.array([0.20, 0.03, 0.78]),
    "placemat_mug":     np.array([0.32, -0.10, 0.78]),
    "placemat_fork":    np.array([0.06, -0.10, 0.78]),
    "placemat_spoon":   np.array([0.06, 0.10, 0.78]),
    "bottle_station":   np.array([0.28, 0.16, 0.80]),
}


def yaw_quat(yaw):
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    return [c, 0, 0, s]


def build_scene():
    spec = mujoco.MjSpec.from_file(WORLD_XML)
    for arm in ARMS:
        arm_spec = mujoco.MjSpec.from_file(SO101_XML)
        arm_spec.meshdir = "."
        spec.attach(arm_spec, prefix=f"{arm}/",
                    frame=spec.worldbody.add_frame(pos=list(BASES[arm]["pos"]),
                                                   quat=yaw_quat(BASES[arm]["yaw"])))
    model = spec.compile()
    return model


def ik_reach(model, arm, target, q_init=None, steps=300, tol=0.005):
    """Damped least-squares IK on the arm's gripperframe site. 5-DoF (pos-only)."""
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/gripperframe")
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}") for j in JOINTS]
    qadr = [model.jnt_qposadr[j] for j in jids]
    if q_init is not None:
        for a, v in zip(qadr, q_init, strict=True):
            data.qpos[a] = v
    for _ in range(steps):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            return True, data.qpos[qadr].copy()
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J = jacp[:, qadr]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        alpha = min(1.0, 0.15 / max(1e-9, np.linalg.norm(err)))
        for i, a in enumerate(qadr):
            lo, hi = model.jnt_range[jids[i]]
            data.qpos[a] = np.clip(data.qpos[a] + alpha * dq[i], lo, hi)
    return False, None


def main():
    model = build_scene()
    print(f"scene compiled: nq={model.nq} nu={model.nu}\n")

    header = f"{'target':18s}" + "".join(f"{a + ' arm':>12s}" for a in ARMS)
    print(header)
    print("-" * len(header))
    reach_map = {}
    for name, tgt in TARGETS.items():
        row = f"{name:18s}"
        for arm in ARMS:
            ok, q = ik_reach(model, arm, tgt)
            reach_map[(name, arm)] = ok
            row += f"{'OK' if ok else 'FAIL':>12s}"
        print(row)

    print("\nReachability summary by zone:")
    zones = {
        "drawer (arm A zone)":   ["drawer_interior", "drawer_front"],
        "handoff (shared)":      ["handoff_centre"],
        "placemat (arm B zone)": ["placemat_plate", "placemat_mug",
                                   "placemat_fork", "placemat_spoon"],
        "bottle station":        ["bottle_station"],
    }
    for zone, names in zones.items():
        a_ok = all(reach_map[(n, "A")] for n in names)
        b_ok = all(reach_map[(n, "B")] for n in names)
        owner = "A" if a_ok and not b_ok else "B" if b_ok and not a_ok else "BOTH" if a_ok and b_ok else "NONE"
        print(f"  {zone:24s} -> owner: {owner}")


if __name__ == "__main__":
    main()
