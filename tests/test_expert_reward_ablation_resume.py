import json
from pathlib import Path

from scripts.expert_reward_ablation_resume import compatible_protocol, verify_or_repair_terminal


def test_runtime_budget_aliases_preserve_frozen_protocol_object():
    original = {
        "id": "EXPERT-REWARD-ABLATION-20261025-D1",
        "updates": 320,
        "budget": {
            "new_environment_steps_per_arm_seed": 5_242_880,
            "new_optimizer_steps_per_arm_seed": 10_240,
        },
        "reward_package": {"minimal": {"delivery": "+2.0 per native capture event"}},
    }

    compatible = compatible_protocol(original)

    assert compatible["budget"]["new_environment_steps_per_seed"] == 5_242_880
    assert compatible["budget"]["new_optimizer_steps_per_seed"] == 10_240
    assert "new_environment_steps_per_seed" not in original["budget"]
    assert "new_optimizer_steps_per_seed" not in original["budget"]
    assert compatible["reward_package"] == original["reward_package"]


def test_completed_run_reconciliation_refuses_nonfinal_or_uncommitted_points(tmp_path, monkeypatch):
    import scripts.expert_reward_ablation_resume as resume

    monkeypatch.setattr(resume, "B", tmp_path / "D")
    monkeypatch.setattr(resume, "R", tmp_path)
    namespace = resume.B
    cp = namespace / "formal/legacy_s2026102851/u000256"
    cp.mkdir(parents=True)
    (namespace / "PROTOCOL.json").write_text(json.dumps({
        "updates": 320,
        "budget": {
            "new_environment_steps_per_arm_seed": 5_242_880,
            "new_optimizer_steps_per_arm_seed": 10_240,
        },
    }))
    (cp / "COMMITTED.json").write_text(json.dumps({
        "passed": True, "new_update": 256,
        "environment_steps": 4_194_304, "new_optimizer_steps": 8_192,
        "state_sha256": "0" * 64, "policy_sha256": "1" * 64,
        "metrics": {"finite": True},
    }))

    assert resume.verify_or_repair_terminal("legacy", 2026102851) is None
    assert not (cp.parent / "COMPLETE.json").exists()


def test_source_keeps_frozen_protocol_and_reward_coefficient_files_untouched():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "runs/expert_reward_20261025/D_ablation_v1/PROTOCOL.json").read_text())
    adapter = (root / "scripts/expert_reward_ablation.py").read_text()
    assert protocol["reward_package"]["minimal"]["delivery"] == "+2.0 per native capture event"
    assert "DELIVERY_REWARD = 2.0" in adapter
    assert "if Path(path).name == \"PROTOCOL.json\":" in \
        (root / "scripts/expert_reward_ablation_resume.py").read_text()
