from __future__ import annotations

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
