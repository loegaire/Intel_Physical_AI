#!/usr/bin/env python3
"""Run honest classical or SmolVLA closed-loop evaluation episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Headless offscreen rendering.  Must be set before ``mujoco`` is imported
# (via envs.dinner_table_env); the default GLFW context returns garbage
# frames at 1280x720 on this machine, which shows up as noise in the videos.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from envs.dinner_table_env import DinnerTableEnv
from perception.perception import PerceptionModule
from policy.orchestrator import Orchestrator


DEFAULT_INSTRUCTION = (
    "Set the table for dinner: open drawer, place utensils, plate, mug, pour water"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class VideoRecorder:
    """Record the actual control trajectory instead of only the final pose."""

    def __init__(self, env: DinnerTableEnv, path: str | None,
                 fps: int = 50, resolution: tuple[int, int] = (1280, 720)):
        self.env = env
        self.path = path
        self.raw_path = None
        self.resolution = resolution
        self.writer = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.raw_path = str(Path(path).with_suffix(".raw.mp4"))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.raw_path, fourcc, fps, resolution)
            if not self.writer.isOpened():
                self.writer = None
                raise RuntimeError(f"Could not open video writer for {self.raw_path}")

    def capture(self, _step: int, _info: dict[str, Any]) -> None:
        if self.writer is None:
            return
        width, height = self.resolution
        frame = self.env.render(camera="overview", height=height, width=width)
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.raw_path and self.path and Path(self.raw_path).exists():
            self._reencode()

    def _reencode(self) -> None:
        """Re-encode raw mp4v to H.264 using ffmpeg for compatibility and size."""
        import subprocess
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", self.raw_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    self.path,
                ],
                check=True,
                timeout=60,
            )
            Path(self.raw_path).unlink(missing_ok=True)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback: keep raw file if ffmpeg fails
            if self.raw_path != self.path:
                Path(self.raw_path).rename(self.path)


def run_demo(
    seed: int = 0,
    video_path: str | None = None,
    video_fps: int = 50,
    video_res: tuple[int, int] = (1280, 720),
    verbose: bool = True,
    policy: str = "classical",
    checkpoint: str = "models/smolvla",
    device: str = "cpu",
    controlled_arm: str = "A",
    max_steps: int = 1000,
    instruction: str = DEFAULT_INSTRUCTION,
    strict_perception: bool = False,
    joint_units: str = "degrees",
    vla_runner: Any | None = None,
) -> dict[str, Any]:
    """Run one episode; ``success`` always means full task completion."""
    env = DinnerTableEnv(seed=seed, randomize=True, render_mode="rgb_array")
    env.reset()
    recorder = VideoRecorder(env, video_path, video_fps, video_res)
    perception = None

    try:
        if verbose:
            print(f"Instruction: {instruction}")
            print(f"Seed: {seed}; policy: {policy}")

        if policy == "smolvla":
            runner = vla_runner
            if runner is None:
                from policy.vla import SmolVLARunner

                runner = SmolVLARunner(
                    checkpoint=checkpoint,
                    device=device,
                    controlled_arm=controlled_arm,
                    joint_units=joint_units,
                )
            policy_result = runner.run_episode(
                env,
                instruction,
                max_steps=max_steps,
                step_callback=recorder.capture,
            )
            result: dict[str, Any] = {
                "seed": seed,
                "policy": policy,
                **policy_result,
            }
        else:
            perception = PerceptionModule(env, update_every=5)
            orchestrator = Orchestrator(
                env,
                perception,
                verbose=verbose,
                step_callback=recorder.capture,
                allow_privileged_fallback=not strict_perception,
            )
            perception.update(force=True)
            perception.update(force=True)
            plan_success, skill_results = orchestrator.execute_instruction(instruction)
            result = {
                "seed": seed,
                "policy": policy,
                "success": bool(env.success()),
                "plan_success": bool(plan_success),
                "steps": orchestrator.executor.total_steps,
                "skill_results": [
                    {"skill": name, "status": status.name}
                    for name, status in skill_results
                ],
                "task_state": env.task_state(),
            }
    finally:
        recorder.close()
        if perception is not None:
            perception.close()
        env.close()

    if video_path and verbose:
        print(f"Video saved to {video_path}")
    return result


def run_multi_seed_demo(
    seeds: list[int],
    output_dir: str,
    video: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate fixed seeds and persist a machine-readable result artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vla_runner = None
    if kwargs.get("policy") == "smolvla":
        from policy.vla import SmolVLARunner

        # Model weights and processors are seed-independent.  Keep one runner
        # for the whole evaluation; run_episode() resets its action queue at
        # every episode boundary.
        vla_runner = SmolVLARunner(
            checkpoint=kwargs.get("checkpoint", "models/smolvla"),
            device=kwargs.get("device", "cpu"),
            controlled_arm=kwargs.get("controlled_arm", "A"),
            joint_units=kwargs.get("joint_units", "degrees"),
        )
    all_results = []
    for seed in seeds:
        video_path = str(output / f"demo_seed_{seed:02d}.mp4") if video else None
        all_results.append(run_demo(
            seed=seed,
            video_path=video_path,
            vla_runner=vla_runner,
            **kwargs,
        ))

    successes = sum(bool(result["success"]) for result in all_results)
    summary = {
        "metric": "full_task_success",
        "seeds": seeds,
        "successes": successes,
        "success_rate": successes / len(seeds) if seeds else 0.0,
        "results": all_results,
    }
    result_path = output / "evaluation.json"
    result_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print("\nFULL-TASK EVALUATION SUMMARY")
    print(f"Seeds: {seeds}")
    print(f"Successes: {successes}/{len(seeds)}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Results: {result_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Physical AI demo/evaluation")
    parser.add_argument("--seed", type=int, default=0, help="First seed")
    parser.add_argument("--seeds", type=int, help="Number of seeds (default: 1, or 10 with --eval-only)")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate 10 seeds by default without video")
    parser.add_argument("--video", nargs="?", const="demo.mp4", help="Record trajectory, optionally to PATH")
    parser.add_argument("--output-dir", default="results", help="Multi-seed artifact directory")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--policy", choices=["classical", "smolvla"], default="classical")
    parser.add_argument("--checkpoint", default="models/smolvla")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--controlled-arm", choices=["A", "B"], default="A")
    parser.add_argument(
        "--joint-units", choices=["degrees", "radians"], default="degrees",
        help="SmolVLA checkpoint joint units (stock SO-100 checkpoints use degrees)",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--strict-perception", action="store_true",
        help="Fail a classical skill when camera perception has no pose estimate",
    )
    args = parser.parse_args()

    count = args.seeds if args.seeds is not None else (10 if args.eval_only else 1)
    common = {
        "verbose": not args.quiet,
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "controlled_arm": args.controlled_arm,
        "max_steps": args.max_steps,
        "instruction": args.instruction,
        "strict_perception": args.strict_perception,
        "joint_units": args.joint_units,
    }
    if count == 1 and not args.eval_only:
        result = run_demo(seed=args.seed, video_path=args.video, **common)
        print(json.dumps(_jsonable(result), indent=2))
    else:
        seeds = list(range(args.seed, args.seed + count))
        run_multi_seed_demo(
            seeds,
            args.output_dir,
            video=bool(args.video) and not args.eval_only,
            **common,
        )


if __name__ == "__main__":
    main()
