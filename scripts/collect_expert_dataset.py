#!/usr/bin/env python3
"""Collect camera/state/action trajectories from the classical expert.

Episodes are stored as compressed NPZ files plus a JSON manifest.  Failed
episodes are rejected by default so they cannot silently become imitation
targets; pass ``--keep-failures`` only for failure-analysis datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from envs.dinner_table_env import DinnerTableEnv
from perception.perception import PerceptionModule
from policy.orchestrator import Orchestrator
from scripts.run_demo import DEFAULT_INSTRUCTION, _jsonable


CAMERAS = ("overhead", "drawer_cam", "placemat_cam")


def collect_episode(seed: int, instruction: str, resolution: int,
                    stride: int, strict_perception: bool) -> tuple[dict, dict]:
    env = DinnerTableEnv(seed=seed, randomize=True, render_mode="rgb_array")
    env.reset()
    frames: dict[str, list[np.ndarray]] = {name: [] for name in CAMERAS}
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []

    def capture(step: int, transition: dict) -> None:
        if step % stride:
            return
        obs = transition["observation"]
        state = np.concatenate((obs["A_qpos"], obs["B_qpos"], [obs["drawer"]]))
        states.append(state.astype(np.float32))
        actions.append(np.asarray(transition["action"], dtype=np.float32))
        rewards.append(float(transition["reward"]))
        for camera in CAMERAS:
            frames[camera].append(
                env.render(camera=camera, height=resolution, width=resolution)
            )

    perception = PerceptionModule(env, update_every=5)
    try:
        orchestrator = Orchestrator(
            env,
            perception,
            verbose=False,
            step_callback=capture,
            allow_privileged_fallback=not strict_perception,
        )
        perception.update(force=True)
        perception.update(force=True)
        plan_success, skill_results = orchestrator.execute_instruction(instruction)
        full_success = bool(env.success())
        final_task_state = env.task_state()
    finally:
        perception.close()
        env.close()
    arrays = {
        "observation.state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        **{
            f"observation.images.{name}": np.asarray(images, dtype=np.uint8)
            for name, images in frames.items()
        },
    }
    metadata = {
        "seed": seed,
        "instruction": instruction,
        "stride": stride,
        "control_hz": 50,
        "sample_hz": 50 / stride,
        "frames": len(states),
        "plan_success": bool(plan_success),
        "success": full_success,
        "metric": "full_task_success",
        "skill_results": [
            {"skill": name, "status": status.name} for name, status in skill_results
        ],
        "task_state": final_task_state,
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect classical expert trajectories")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="First seed")
    parser.add_argument("--output-dir", default="data/expert")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--strict-perception", action="store_true")
    parser.add_argument("--keep-failures", action="store_true")
    args = parser.parse_args()

    if args.seeds < 1 or args.stride < 1 or args.resolution < 16:
        parser.error("--seeds/--stride must be positive and --resolution must be >= 16")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in range(args.seed, args.seed + args.seeds):
        arrays, metadata = collect_episode(
            seed, args.instruction, args.resolution, args.stride,
            args.strict_perception,
        )
        accepted = bool(metadata["success"] or args.keep_failures)
        metadata["accepted"] = accepted
        if accepted:
            filename = f"episode_{seed:04d}.npz"
            np.savez_compressed(output / filename, **arrays)
            metadata["file"] = filename
        manifest.append(_jsonable(metadata))
        print(
            f"seed={seed} success={metadata['success']} frames={metadata['frames']} "
            f"saved={accepted}"
        )

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    accepted_count = sum(bool(item["accepted"]) for item in manifest)
    print(f"Accepted episodes: {accepted_count}/{len(manifest)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
