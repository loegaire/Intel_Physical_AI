import numpy as np

from envs.dinner_table_env import COUNTERTOP_Z, DinnerTableEnv


def test_scripted_place_object_updates_task_state_and_reset_clears_it():
    env = DinnerTableEnv(seed=0, randomize=False)
    env.reset()

    assert not env.task_state()["plate_placed"]

    env.scripted_place_object("plate", (0.25, 0.03), COUNTERTOP_Z + 0.02)
    state = env.task_state()

    assert state["plate_placed"]
    np.testing.assert_allclose(state["plate_pos"], [0.25, 0.03, COUNTERTOP_Z + 0.02])

    env.reset()
    assert not env.task_state()["plate_placed"]


def test_success_requires_every_place_setting_and_pour():
    env = DinnerTableEnv(seed=0, randomize=False)
    env.reset()

    for name, xy in {
        "plate": (0.25, 0.03),
        "mug": (0.31, -0.10),
        "fork": (0.11, -0.10),
        "spoon": (0.11, 0.10),
    }.items():
        env.scripted_place_object(name, xy, COUNTERTOP_Z + 0.02)

    assert not env.success()
    env._poured = True
    assert env.success()
