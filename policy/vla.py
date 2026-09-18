"""Closed-loop SmolVLA inference for the dinner-table environment.

The module deliberately keeps the simulator/action contract separate from
LeRobot.  This makes the safety-critical part (mapping a policy action onto
the two arms) testable without loading a large checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np


CAMERA_ORDER = ("overhead", "drawer_cam", "placemat_cam")
SUPPORTED_ACTION_DIMS = (6, 12, 13)


class VLAContractError(ValueError):
    """Raised when a checkpoint cannot be mapped to the simulator contract."""


def _feature_type(feature: Any) -> str:
    value = feature.get("type") if isinstance(feature, dict) else getattr(feature, "type", "")
    return str(getattr(value, "value", value)).upper()


def _feature_shape(feature: Any) -> tuple[int, ...]:
    value = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", ())
    return tuple(int(v) for v in value)


def _input_features(config: Any) -> dict[str, Any]:
    features = getattr(config, "input_features", None)
    if not isinstance(features, dict):
        raise VLAContractError("Checkpoint config has no input_features mapping")
    return features


def action_dim_from_config(config: Any) -> int:
    outputs = getattr(config, "output_features", None)
    if not isinstance(outputs, dict) or "action" not in outputs:
        raise VLAContractError("Checkpoint config has no action output feature")
    shape = _feature_shape(outputs["action"])
    if len(shape) != 1 or shape[0] not in SUPPORTED_ACTION_DIMS:
        raise VLAContractError(
            f"Unsupported action shape {shape}; expected one of (6,), (12,), (13,)"
        )
    return shape[0]


def state_dim_from_config(config: Any) -> int:
    states = [
        _feature_shape(feature)
        for feature in _input_features(config).values()
        if "STATE" in _feature_type(feature)
    ]
    if len(states) != 1 or len(states[0]) != 1:
        raise VLAContractError(f"Expected one vector state feature, found {states}")
    if states[0][0] not in SUPPORTED_ACTION_DIMS:
        raise VLAContractError(
            f"Unsupported state shape {states[0]}; expected 6, 12, or 13 values"
        )
    return states[0][0]


def image_features_from_config(config: Any) -> list[tuple[str, tuple[int, ...]]]:
    images = [
        (name, _feature_shape(feature))
        for name, feature in _input_features(config).items()
        if "VISUAL" in _feature_type(feature)
    ]
    if not images:
        raise VLAContractError("Checkpoint config has no visual input features")
    if len(images) > len(CAMERA_ORDER):
        raise VLAContractError(
            f"Checkpoint needs {len(images)} cameras but the environment provides {len(CAMERA_ORDER)}"
        )
    for name, shape in images:
        if len(shape) != 3 or shape[0] != 3:
            raise VLAContractError(f"Visual feature {name!r} must be CHW RGB, got {shape}")
    return images


def _convert_arm_units(values: np.ndarray, joint_units: str, to_radians: bool) -> np.ndarray:
    if joint_units not in ("degrees", "radians"):
        raise VLAContractError("joint_units must be 'degrees' or 'radians'")
    if joint_units == "radians":
        return values
    return np.deg2rad(values) if to_radians else np.rad2deg(values)


def simulator_state(
    env: Any,
    dimension: int,
    controlled_arm: str = "A",
    joint_units: str = "radians",
) -> np.ndarray:
    """Return proprioception matching a 6D, 12D, or 13D checkpoint."""
    if controlled_arm not in ("A", "B"):
        raise VLAContractError("controlled_arm must be 'A' or 'B'")
    obs = env._get_obs()
    arm_a = np.asarray(obs["A_qpos"], dtype=np.float32)
    arm_b = np.asarray(obs["B_qpos"], dtype=np.float32)
    if dimension == 6:
        state = (arm_a if controlled_arm == "A" else arm_b).copy()
        return _convert_arm_units(state, joint_units, to_radians=False)
    state = np.concatenate((arm_a, arm_b))
    state = _convert_arm_units(state, joint_units, to_radians=False)
    if dimension == 13:
        state = np.concatenate((state, np.asarray([obs["drawer"]], dtype=np.float32)))
    if state.shape != (dimension,):
        raise VLAContractError(f"Built state {state.shape}, expected ({dimension},)")
    return state.astype(np.float32, copy=False)


class ActionAdapter:
    """Map a post-processed policy action to the environment's 13 controls.

    A stock 6D SmolVLA checkpoint is explicitly single-arm: the other arm is
    held at its current joint position.  Only 12D/13D checkpoints are treated
    as bimanual.
    """

    def __init__(self, controlled_arm: str = "A", joint_units: str = "radians"):
        if controlled_arm not in ("A", "B"):
            raise VLAContractError("controlled_arm must be 'A' or 'B'")
        self.controlled_arm = controlled_arm
        if joint_units not in ("degrees", "radians"):
            raise VLAContractError("joint_units must be 'degrees' or 'radians'")
        self.joint_units = joint_units

    def to_env(self, action: Any, observation: dict[str, Any]) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float64).squeeze()
        if raw.ndim != 1 or raw.size not in SUPPORTED_ACTION_DIMS:
            raise VLAContractError(
                f"Policy returned shape {np.asarray(action).shape}; expected 6, 12, or 13 values"
            )
        if not np.isfinite(raw).all():
            raise VLAContractError("Policy action contains NaN or infinite values")

        arm_values = raw[:12] if raw.size >= 12 else raw
        converted = _convert_arm_units(arm_values, self.joint_units, to_radians=True)
        if raw.size == 13:
            raw = np.concatenate((converted, raw[12:13]))
        else:
            raw = converted

        if raw.size == 13:
            return raw.copy()
        if raw.size == 12:
            return np.concatenate((raw, np.zeros(1, dtype=np.float64)))

        arm_a = np.asarray(observation["A_qpos"], dtype=np.float64).copy()
        arm_b = np.asarray(observation["B_qpos"], dtype=np.float64).copy()
        if self.controlled_arm == "A":
            arm_a = raw
        else:
            arm_b = raw
        return np.concatenate((arm_a, arm_b, np.zeros(1, dtype=np.float64)))


def build_lerobot_frame(
    env: Any,
    config: Any,
    instruction: str,
    controlled_arm: str = "A",
    joint_units: str = "radians",
) -> dict[str, Any]:
    """Build one unbatched LeRobot frame from camera-only observations."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for SmolVLA inference") from exc

    frame: dict[str, Any] = {
        "observation.state": torch.from_numpy(
            simulator_state(
                env, state_dim_from_config(config), controlled_arm, joint_units
            )
        ),
        "task": instruction,
    }
    for camera_name, (feature_name, shape) in zip(
        CAMERA_ORDER, image_features_from_config(config), strict=False
    ):
        _, height, width = shape
        rgb = env.render(camera=camera_name, height=height, width=width)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        frame[feature_name] = torch.from_numpy(chw)
    return frame


@dataclass
class VLAStep:
    step: int
    inference_ms: float
    reward: float
    done: bool


class SmolVLARunner:
    """Lazy-loaded LeRobot SmolVLA policy with closed-loop execution."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        controlled_arm: str = "A",
        joint_units: str = "degrees",
    ):
        self.checkpoint = str(checkpoint)
        self.device = device
        self.adapter = ActionAdapter(controlled_arm, joint_units)
        self.controlled_arm = controlled_arm
        self.joint_units = joint_units
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "SmolVLA dependencies are incomplete. Install `lerobot[smolvla]` "
                "and a torchvision build compatible with the installed PyTorch."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")

        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.action_dim = action_dim_from_config(self.policy.config)
        self.bimanual = self.action_dim in (12, 13)
        self._torch = torch

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, env: Any, instruction: str) -> tuple[np.ndarray, float]:
        frame = build_lerobot_frame(
            env, self.policy.config, instruction, self.controlled_arm,
            self.joint_units,
        )
        batch = self.preprocess(frame)
        start = perf_counter()
        with self._torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocess(action)
        inference_ms = (perf_counter() - start) * 1000.0
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        return np.asarray(action).squeeze(), inference_ms

    def run_episode(
        self,
        env: Any,
        instruction: str,
        max_steps: int = 1000,
        step_callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self.reset()
        observation = env._get_obs()
        trace: list[VLAStep] = []
        total_reward = 0.0
        for step in range(max_steps):
            policy_action, inference_ms = self.predict(env, instruction)
            env_action = self.adapter.to_env(policy_action, observation)
            observation, reward, done, info = env.step(env_action)
            total_reward += float(reward)
            trace.append(VLAStep(step, inference_ms, float(reward), bool(done)))
            if step_callback is not None:
                transition = dict(info)
                transition.update({"reward": float(reward), "done": bool(done)})
                step_callback(step, transition)
            if done:
                break

        latencies = np.asarray([item.inference_ms for item in trace], dtype=float)
        return {
            "success": bool(env.success()),
            "steps": len(trace),
            "total_reward": total_reward,
            "action_dim": self.action_dim,
            "bimanual": self.bimanual,
            "controlled_arm": self.controlled_arm,
            "joint_units": self.joint_units,
            "inference_ms": {
                "mean": float(latencies.mean()) if latencies.size else None,
                "p50": float(np.percentile(latencies, 50)) if latencies.size else None,
                "p90": float(np.percentile(latencies, 90)) if latencies.size else None,
                "p99": float(np.percentile(latencies, 99)) if latencies.size else None,
            },
            "task_state": env.task_state(),
        }
