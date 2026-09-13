"""Perception accuracy check: camera-based estimates vs simulator state.

Runs the RGB-D detector over several randomized seeds (drawer CLOSED and
OPEN states) and reports per-object world-frame error in millimetres.

Run:  python scripts/check_perception.py [n_seeds]
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.dinner_table_env import DinnerTableEnv  # noqa: E402
from perception.perception import PerceptionModule  # noqa: E402

OBJECTS = ["plate", "mug", "bottle", "spoon", "fork"]


def run_seed(seed: int, open_drawer: bool) -> dict[str, float]:
    env = DinnerTableEnv(seed=seed, randomize=True)
    env.reset()
    if open_drawer:
        env.data.qpos[env._drawer_qadr] = 0.26
        import mujoco
        mujoco.mj_forward(env.model, env.data)

    perc = PerceptionModule(env, resolution=448, update_every=1)
    # two updates so the EMA filter settles on the first detection
    perc.update(force=True)
    perc.update(force=True)

    errs: dict[str, float] = {}
    for name in OBJECTS:
        est = perc.object_pos(name)
        true = env.object_pos(name)
        if est is None:
            errs[name] = float("nan")
        else:
            errs[name] = float(np.linalg.norm(est - true) * 1000.0)  # mm
    errs["__drawer"] = (abs(perc.drawer_qpos() - env.data.qpos[env._drawer_qadr])
                        * 1000.0 if perc.drawer_qpos() is not None
                        else float("nan"))
    return errs


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    names = OBJECTS + ["__drawer"]
    print(f"{'seed':>6} {'state':>7} " +
          " ".join(f"{n:>8}" for n in names))
    agg: dict[str, list[float]] = {n: [] for n in names}
    for seed in range(n_seeds):
        for state in ("closed", "open"):
            errs = run_seed(seed, state == "open")
            print(f"{seed:>6} {state:>7} " +
                  " ".join(f"{errs[n]:8.1f}" for n in names))
            for n in names:
                if np.isfinite(errs[n]):
                    agg[n].append(errs[n])
    print("\nsummary (mean err, mm):")
    for n in names:
        vals = agg[n]
        if vals:
            print(f"  {n:>8}: mean {np.mean(vals):6.1f}  max {np.max(vals):7.1f}")


if __name__ == "__main__":
    main()
