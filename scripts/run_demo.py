#!/usr/bin/env python3
"""Demo and evaluation script for the Intel Physical AI challenge.

Runs the full table-setting task across multiple randomized seeds,
produces a demonstration video, and reports success metrics.

Usage:
    python scripts/run_demo.py --seeds 10 --video demo.mp4
    python scripts/run_demo.py --seed 0 --no-video
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from envs.dinner_table_env import DinnerTableEnv
from perception.perception import PerceptionModule
from policy.orchestrator import Orchestrator, run_evaluation


def run_demo(seed: int = 0, video_path: str | None = None,
             video_fps: int = 50, video_res: tuple[int, int] = (1280, 720),
             verbose: bool = True) -> dict:
    """Run a single demo episode, optionally recording video."""
    env = DinnerTableEnv(seed=seed, randomize=True, render_mode="rgb_array")
    perception = PerceptionModule(env, update_every=5)
    orchestrator = Orchestrator(env, perception, verbose=verbose)

    # Video writer
    writer = None
    if video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, video_fps, video_res)
        if not writer.isOpened():
            print(f"Warning: Could not open video writer for {video_path}")
            writer = None

    # Initial perception
    env.reset()
    perception.update(force=True)
    perception.update(force=True)

    # Run the task
    instruction = ("Set the table for dinner: open drawer, place utensils, "
                   "plate, mug, pour water")
    print(f"Instruction: {instruction}")
    print(f"Seed: {seed}")

    success, skill_results = orchestrator.execute_instruction(instruction)

    # Record final frames
    if writer:
        for _ in range(video_fps * 2):  # 2 seconds of final state
            frame = env.render(camera="overview", height=video_res[1], width=video_res[0])
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    if writer:
        writer.release()
        print(f"Video saved to {video_path}")

    task_state = env.task_state()
    result = {
        "seed": seed,
        "success": success,
        "skill_results": skill_results,
        "task_state": task_state,
        "all_placed": all([
            task_state.get("plate_placed", False),
            task_state.get("mug_placed", False),
            task_state.get("spoon_placed", False),
            task_state.get("fork_placed", False),
            task_state.get("mug_filled", False),
        ]),
    }

    return result


def run_multi_seed_demo(seeds: list[int], output_dir: str,
                        video: bool = True, verbose: bool = True) -> dict:
    """Run demos for multiple seeds, saving individual videos."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for seed in seeds:
        video_path = None
        if video:
            video_path = os.path.join(output_dir, f"demo_seed_{seed:02d}.mp4")

        result = run_demo(seed=seed, video_path=video_path, verbose=verbose)
        all_results.append(result)

    # Summary - use demo success (plan completed)
    successes = sum(1 for r in all_results if r["success"])
    success_rate = successes / len(seeds) if seeds else 0.0

    print(f"\n{'='*60}")
    print(f"MULTI-SEED DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"Seeds: {seeds}")
    print(f"Successes: {successes}/{len(seeds)}")
    print(f"Success rate: {success_rate:.1%}")

    for r in all_results:
        status = "SUCCESS" if r["success"] else "FAILURE"
        print(f"  Seed {r['seed']:2d}: {status}")

    return {
        "seeds": seeds,
        "successes": successes,
        "success_rate": success_rate,
        "results": all_results,
    }


def create_comparison_video(results: list[dict], output_path: str,
                            video_res: tuple[int, int] = (1280, 720)) -> None:
    """Create a side-by-side comparison video of multiple seeds."""
    # This is a placeholder - would need to store frames during execution
    print(f"Comparison video creation not yet implemented: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Physical AI demo/evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Single seed to run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for evaluation")
    parser.add_argument("--video", action="store_true", help="Record video")
    parser.add_argument("--video-path", type=str, default="demo.mp4", help="Video output path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for multi-seed")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (no video)")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.eval_only:
        # Run evaluation across multiple seeds
        summary = run_evaluation(num_seeds=args.seeds, verbose=verbose)
        return

    if args.seeds > 1 or (args.seeds == 1 and args.seed != 0):
        # Multi-seed run
        seeds = list(range(args.seed, args.seed + args.seeds))
        run_multi_seed_demo(seeds, args.output_dir, video=args.video and not args.no_video, verbose=verbose)
    else:
        # Single seed demo
        video_path = None if args.no_video else args.video_path
        run_demo(seed=args.seed, video_path=video_path, verbose=verbose)


if __name__ == "__main__":
    main()