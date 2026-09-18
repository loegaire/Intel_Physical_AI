from __future__ import annotations

import numpy as np

import scripts.run_demo as demo


def test_multi_seed_reuses_one_smolvla_runner(monkeypatch, tmp_path):
    created = []
    runners_seen = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    def fake_run_demo(*, seed, video_path, vla_runner, **kwargs):
        del video_path, kwargs
        runners_seen.append(vla_runner)
        return {"seed": seed, "success": False}

    monkeypatch.setattr("policy.vla.SmolVLARunner", FakeRunner)
    monkeypatch.setattr(demo, "run_demo", fake_run_demo)

    result = demo.run_multi_seed_demo(
        [3, 4, 5],
        str(tmp_path),
        video=False,
        policy="smolvla",
        checkpoint="models/test-checkpoint",
        device="cpu",
        controlled_arm="A",
        joint_units="degrees",
    )

    assert len(created) == 1
    assert runners_seen == [created[0], created[0], created[0]]
    assert result["seeds"] == [3, 4, 5]


def test_video_overlay_burns_command_and_seed_into_frame(monkeypatch):
    class FakeEnv:
        def task_state(self):
            return {
                "drawer_open": True,
                "plate_placed": True,
                "mug_placed": False,
                "fork_placed": False,
                "spoon_placed": False,
                "mug_filled": False,
            }

        def success(self):
            return False

    frame = np.zeros((160, 480, 3), dtype=np.uint8)
    recorder = demo.VideoRecorder(
        FakeEnv(),
        path=None,
        overlay={"instruction": "Set the table", "seed": 7, "policy": "classical"},
    )

    recorder._draw_overlay(frame, 12, {"reward": 1.5})

    assert frame.sum() > 0
