"""Orchestrator: natural-language task planner and skill sequencer.

The orchestrator bridges the gap between high-level task descriptions
(e.g. "set the table for dinner") and the low-level skill layer.

Architecture:
- LLM/VLM planner (stubbed) -> structured plan (list of skill specs)
- Skill executor runs each skill to completion, feeding perception
- Perception provides object poses; skills consume via env pose provider
- State machine tracks task progress and handles retries

For the Intel Physical AI challenge, the planner can be replaced by a
VLA policy (SmolVLA, Pi0.5, ACT) that outputs actions directly.
This orchestrator provides a classical fallback and evaluation harness.
"""

from __future__ import annotations

import enum
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from envs.dinner_table_env import DinnerTableEnv, ARM_BASES, COUNTERTOP_Z
from perception.perception import PerceptionModule
from policy.skills import (
    Skill, SkillStatus, MoveTo, Grasp, Place, OpenDrawer, Pour,
    GRIP_OPEN, GRIP_CLOSED,
)


@dataclass
class SkillSpec:
    """Specification for one skill invocation."""
    name: str
    params: dict
    max_retries: int = 1


@dataclass
class Plan:
    """A sequence of skills with metadata."""
    steps: list[SkillSpec]
    description: str = ""


class Planner:
    """Planner that converts natural language to skill sequences.

    This is a rule-based stub. In a full VLA system, this would be
    replaced by a vision-language model that outputs structured plans
    or directly outputs actions.
    """

    # Goal positions for place skills (world frame xy)
    PLACE_GOALS = {
        "plate": (0.25, 0.03),
        "mug": (0.31, -0.10),
        "fork": (0.11, -0.10),
        "spoon": (0.11, 0.10),
    }

    def __init__(self, env: DinnerTableEnv):
        self.env = env

    def parse_instruction(self, instruction: str) -> Plan:
        """Parse a natural-language instruction into a skill plan."""
        instruction = instruction.lower().strip()

        # Pre-defined task templates
        if "set the table" in instruction or "dinner table" in instruction:
            return self._plan_full_table_setting()
        if "open drawer" in instruction:
            return self._plan_open_drawer()
        if "pour" in instruction:
            return self._plan_pour()
        if "pick up" in instruction or "grasp" in instruction:
            return self._plan_pick_place(instruction)
        if "place" in instruction or "put" in instruction:
            return self._plan_pick_place(instruction)

        # Fallback: try to extract objects and actions
        return self._plan_generic(instruction)

    def _plan_full_table_setting(self) -> Plan:
        """Complete dinner table setting sequence.

        Note: This classical skill-based approach is a reference implementation.
        The VLA policy (SmolVLA, Pi0.5, ACT) would replace this with an 
        end-to-end learned policy in the full challenge solution.
        
        This is the complete classical expert plan.  It is also the source of
        demonstrations used to fine-tune a learned policy; individual skill
        failures remain visible in the per-step result list.
        """
        steps = [
            SkillSpec("OpenDrawer", {"arm": "A"}, max_retries=1),
            SkillSpec("Grasp", {"arm": "A", "obj_name": "spoon"}, max_retries=1),
            SkillSpec("Place", {"arm": "A", "obj_name": "spoon",
                                "goal_xy": self.PLACE_GOALS["spoon"]}, max_retries=1),
            SkillSpec("Grasp", {"arm": "A", "obj_name": "fork"}, max_retries=1),
            SkillSpec("Place", {"arm": "A", "obj_name": "fork",
                                "goal_xy": self.PLACE_GOALS["fork"]}, max_retries=1),
            SkillSpec("Grasp", {"arm": "B", "obj_name": "plate"}, max_retries=1),
            SkillSpec("Place", {"arm": "B", "obj_name": "plate",
                                "goal_xy": self.PLACE_GOALS["plate"]}, max_retries=1),
            SkillSpec("Grasp", {"arm": "B", "obj_name": "mug"}, max_retries=1),
            SkillSpec("Place", {"arm": "B", "obj_name": "mug",
                                "goal_xy": self.PLACE_GOALS["mug"]}, max_retries=1),
            SkillSpec("Grasp", {"arm": "A", "obj_name": "bottle"}, max_retries=1),
            SkillSpec("Pour", {"arm": "A", "hold_arm": "B"}, max_retries=1),
        ]
        return Plan(steps=steps, description="Complete classical table-setting expert plan")

    def _plan_open_drawer(self) -> Plan:
        return Plan(steps=[SkillSpec("OpenDrawer", {"arm": "A"})],
                    description="Open the drawer")

    def _plan_pour(self) -> Plan:
        return Plan(steps=[
            SkillSpec("Grasp", {"arm": "A", "obj_name": "bottle"}),
            SkillSpec("Pour", {"arm": "A", "hold_arm": "B"}),
        ], description="Pour from bottle into mug")

    def _plan_pick_place(self, instruction: str) -> Plan:
        """Extract object and target from pick/place instruction."""
        # Simple keyword matching
        obj = None
        for name in ["plate", "mug", "bottle", "spoon", "fork"]:
            if name in instruction:
                obj = name
                break

        arm = "A" if "left" in instruction or "arm a" in instruction else "B"

        steps = []
        if "pick" in instruction or "grasp" in instruction or "pick up" in instruction:
            if obj:
                steps.append(SkillSpec("Grasp", {"arm": arm, "obj_name": obj}))
        if "place" in instruction or "put" in instruction:
            if obj and obj in self.PLACE_GOALS:
                steps.append(SkillSpec("Place", {"arm": arm, "obj_name": obj,
                                                 "goal_xy": self.PLACE_GOALS[obj]}))

        return Plan(steps=steps, description=f"Pick/place {obj} with arm {arm}")

    def _plan_generic(self, instruction: str) -> Plan:
        """Fallback generic planner."""
        return Plan(steps=[], description=f"Unrecognized instruction: {instruction}")


class SkillExecutor:
    """Executes a skill to completion, with perception integration."""

    def __init__(self, env: DinnerTableEnv, perception: PerceptionModule,
                 max_skill_steps: int = 1000,
                 step_callback: Callable[[int, dict], None] | None = None):
        self.env = env
        self.perception = perception
        self.max_skill_steps = max_skill_steps
        self.step_callback = step_callback
        self.total_steps = 0

    def run_skill(self, skill: Skill, verbose: bool = False) -> SkillStatus:
        """Run a skill until completion or timeout."""
        # Per-skill step limits
        skill_limits = {
            "OpenDrawer": 500,
            "Grasp": 500,
            "Place": 500,
            "Pour": 500,
            "MoveTo": 300,
        }
        limit = skill_limits.get(skill.__class__.__name__, self.max_skill_steps)

        for step in range(limit):
            try:
                action = skill.act()
            except LookupError as exc:
                skill.status = SkillStatus.FAILURE
                if verbose:
                    print(f"  Skill failed: {exc}")
                return SkillStatus.FAILURE
            obs, reward, done, info = self.env.step(action)
            if self.step_callback is not None:
                transition = dict(info)
                transition.update({
                    "observation": obs,
                    "action": np.asarray(action).copy(),
                    "reward": float(reward),
                    "done": bool(done),
                })
                self.step_callback(self.total_steps, transition)
            self.total_steps += 1

            # Update perception periodically
            if step % 10 == 0:
                self.perception.update()

            if verbose and step % 50 == 0:
                print(f"  Step {step}: phase={skill.phase if hasattr(skill, 'phase') else 'N/A'}, "
                      f"status={skill.status.name}")

            if skill.status != SkillStatus.RUNNING:
                if verbose:
                    print(f"  Skill finished: {skill.status.name} at step {step}")
                return skill.status

        # Timeout
        skill.status = SkillStatus.FAILURE
        if verbose:
            print(f"  Skill timeout after {self.max_skill_steps} steps")
        return SkillStatus.FAILURE


class Orchestrator:
    """Main orchestrator: plan + execute + monitor."""

    def __init__(self, env: DinnerTableEnv, perception: Optional[PerceptionModule] = None,
                 verbose: bool = True,
                 step_callback: Callable[[int, dict], None] | None = None,
                 allow_privileged_fallback: bool = True):
        self.env = env
        self.verbose = verbose
        self.planner = Planner(env)
        self.perception = perception or PerceptionModule(env, update_every=5)
        self.executor = SkillExecutor(env, self.perception,
                                      step_callback=step_callback)
        self._skill_map = {
            "MoveTo": MoveTo,
            "Grasp": Grasp,
            "Place": Place,
            "OpenDrawer": OpenDrawer,
            "Pour": Pour,
        }

        # Connect perception to env for skill consumption
        self.env.set_pose_provider(
            self._perception_provider,
            allow_fallback=allow_privileged_fallback,
        )

    def _perception_provider(self, name: str):
        """Provide object poses from perception to skills."""
        if name == "__handle":
            pos = self.perception.handle_pos()
            return (pos, None) if pos is not None else None
        pos = self.perception.object_pos(name)
        yaw = self.perception.object_yaw(name)
        return (pos, yaw) if pos is not None else None

    def _create_skill(self, spec: SkillSpec) -> Skill:
        """Instantiate a skill from its spec."""
        cls = self._skill_map.get(spec.name)
        if cls is None:
            raise ValueError(f"Unknown skill: {spec.name}")
        return cls(self.env, **spec.params)

    def execute_plan(self, plan: Plan) -> tuple[bool, list[tuple[str, SkillStatus]]]:
        """Execute a full plan, returning overall success and per-skill results."""
        results = []
        overall_success = True

        if self.verbose:
            print(f"\nExecuting plan: {plan.description}")
            print(f"Steps: {len(plan.steps)}")

        for i, spec in enumerate(plan.steps):
            if self.verbose:
                print(f"\n[{i+1}/{len(plan.steps)}] {spec.name}({spec.params})")

            status = SkillStatus.FAILURE

            for attempt in range(spec.max_retries + 1):
                if attempt > 0 and self.verbose:
                    print(f"  Retry {attempt}/{spec.max_retries}")
                # A failed finite-state skill cannot be reused: its status,
                # phase timers, IK caches and contact history are terminal.
                skill = self._create_skill(spec)
                status = self.executor.run_skill(skill, verbose=self.verbose)
                if status == SkillStatus.SUCCESS:
                    break

            results.append((spec.name, status))
            if status != SkillStatus.SUCCESS:
                overall_success = False
                if self.verbose:
                    print(f"  FAILED: {spec.name}")
                # Continue with plan (don't abort on single failure)

        if self.verbose:
            print(f"\nPlan complete. Overall: {'SUCCESS' if overall_success else 'FAILURE'}")
            for name, status in results:
                print(f"  {name}: {status.name}")

        return overall_success, results

    def execute_instruction(self, instruction: str) -> tuple[bool, list[tuple[str, SkillStatus]]]:
        """Parse and execute a natural language instruction."""
        plan = self.planner.parse_instruction(instruction)
        return self.execute_plan(plan)

    def run_full_demo(self, seed: int = 0) -> dict:
        """Run the full table-setting demo for one seed."""
        self.env.reset(seed=seed)
        self.perception.reset()

        # Initial perception update
        self.perception.update(force=True)
        self.perception.update(force=True)

        instruction = "Set the table for dinner: open drawer, place utensils, plate, mug, pour water"
        success, results = self.execute_instruction(instruction)

        task_state = self.env.task_state()
        full_task_success = bool(self.env.success())
        return {
            "seed": seed,
            "success": full_task_success,
            "plan_success": success,
            "skill_results": results,
            "task_state": task_state,
            "all_placed": all([
                task_state.get("plate_placed", False),
                task_state.get("mug_placed", False),
                task_state.get("spoon_placed", False),
                task_state.get("fork_placed", False),
                task_state.get("mug_filled", False),
            ]),
            "drawer_open": task_state.get("drawer_open", False),
        }


def run_evaluation(num_seeds: int = 10, verbose: bool = True) -> dict:
    """Run evaluation across multiple randomized seeds."""
    results = []
    successes = 0

    for seed in range(num_seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"SEED {seed}")
            print(f"{'='*60}")

        env = DinnerTableEnv(seed=seed, randomize=True)
        perception = PerceptionModule(env, update_every=5)
        orchestrator = Orchestrator(env, perception, verbose=verbose)

        result = orchestrator.run_full_demo(seed)
        results.append(result)

        if result["success"]:
            successes += 1
            if verbose:
                print(f"Seed {seed}: SUCCESS")
        else:
            if verbose:
                print(f"Seed {seed}: FAILURE")
                print(f"  Task state: {result['task_state']}")

    success_rate = successes / num_seeds if num_seeds > 0 else 0.0

    summary = {
        "num_seeds": num_seeds,
        "successes": successes,
        "success_rate": success_rate,
        "results": results,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Seeds: {num_seeds}")
        print(f"Successes: {successes}")
        print(f"Success rate: {success_rate:.1%}")

    return summary


if __name__ == "__main__":
    # Quick test
    env = DinnerTableEnv(seed=0, randomize=False)
    perception = PerceptionModule(env, update_every=5)
    orchestrator = Orchestrator(env, perception, verbose=True)

    result = orchestrator.run_full_demo(seed=0)
    print(f"\nResult: {result}")
