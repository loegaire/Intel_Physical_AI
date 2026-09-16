from types import SimpleNamespace

import numpy as np
import pytest

from policy.vla import (
    ActionAdapter,
    VLAContractError,
    action_dim_from_config,
    image_features_from_config,
    simulator_state,
    state_dim_from_config,
)


class FakeEnv:
    def _get_obs(self):
        return {
            "A_qpos": np.arange(6, dtype=np.float32),
            "B_qpos": np.arange(10, 16, dtype=np.float32),
            "drawer": np.float32(0.25),
        }


def config(state_dim=6, action_dim=6, cameras=3):
    inputs = {
        "observation.state": {"type": "STATE", "shape": [state_dim]},
    }
    for index in range(cameras):
        inputs[f"observation.images.camera{index + 1}"] = {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        }
    return SimpleNamespace(
        input_features=inputs,
        output_features={"action": {"type": "ACTION", "shape": [action_dim]}},
    )


def test_six_dimensional_action_controls_only_selected_arm():
    observation = FakeEnv()._get_obs()
    mapped = ActionAdapter("B").to_env(np.arange(20, 26), observation)
    np.testing.assert_array_equal(mapped[:6], observation["A_qpos"])
    np.testing.assert_array_equal(mapped[6:12], np.arange(20, 26))
    assert mapped[12] == 0


def test_twelve_dimensional_action_is_bimanual_and_drawer_neutral():
    raw = np.arange(12)
    mapped = ActionAdapter().to_env(raw, FakeEnv()._get_obs())
    np.testing.assert_array_equal(mapped[:12], raw)
    assert mapped[12] == 0


def test_thirteen_dimensional_action_is_preserved():
    raw = np.arange(13)
    np.testing.assert_array_equal(ActionAdapter().to_env(raw, FakeEnv()._get_obs()), raw)


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(7), [0, 1, np.nan, 3, 4, 5]])
def test_invalid_policy_actions_fail_closed(bad):
    with pytest.raises(VLAContractError):
        ActionAdapter().to_env(bad, FakeEnv()._get_obs())


def test_state_contracts_are_explicit():
    env = FakeEnv()
    np.testing.assert_array_equal(simulator_state(env, 6, "A"), np.arange(6))
    np.testing.assert_array_equal(simulator_state(env, 6, "B"), np.arange(10, 16))
    assert simulator_state(env, 12).shape == (12,)
    assert simulator_state(env, 13)[-1] == pytest.approx(0.25)
    np.testing.assert_allclose(
        simulator_state(env, 6, "A", "degrees"), np.rad2deg(np.arange(6))
    )


def test_degree_actions_are_converted_to_mujoco_radians():
    mapped = ActionAdapter("A", "degrees").to_env(
        [0, 90, -90, 180, -180, 45], FakeEnv()._get_obs()
    )
    np.testing.assert_allclose(
        mapped[:6], [0, np.pi / 2, -np.pi / 2, np.pi, -np.pi, np.pi / 4]
    )


def test_checkpoint_contract_introspection():
    cfg = config(state_dim=12, action_dim=13)
    assert state_dim_from_config(cfg) == 12
    assert action_dim_from_config(cfg) == 13
    assert len(image_features_from_config(cfg)) == 3


def test_more_than_three_cameras_is_rejected():
    with pytest.raises(VLAContractError):
        image_features_from_config(config(cameras=4))
