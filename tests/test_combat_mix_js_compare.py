import json
import subprocess

from scripts.combat_mix_js_compare import ROOT


def test_compare_uses_real_production_javascript_and_fixed_modes():
    source = (ROOT / "scripts/combat_mix_js_compare.py").read_text()
    assert '"scripts/je05_web_hunter.js"' in source
    assert '"scripts/expert_validation_js_eval.js"' in source
    assert '("categorical", "greedy")' in source
    assert 'teacher=False' in source


def test_web_manifest_identity_hashes_match_local_exact_checkpoints():
    manifest = json.loads((ROOT / "web/models.json").read_text())
    expected = {
        "bun392-selfplay3072": "runs/je03_bun_stage2_20261012/bun392_selfplay12_v2/formal/bun392_selfplay_dynamic/gen003072/policy.pt",
        "bun392-transport1536": "runs/je05_bun_20261013/combat_transport_pfsp8_v1/formal/bun392_combat_transport_pfsp8/gen001536/policy.pt",
    }
    rows = {row["id"]: row for row in manifest["models"]}
    for model, checkpoint in expected.items():
        assert rows[model]["onnx"]["meta"]["checkpoint_sha256"]
        result = subprocess.run(["sha256sum", checkpoint], cwd=ROOT, check=True,
                                capture_output=True, text=True)
        assert result.stdout.split()[0] == rows[model]["onnx"]["meta"]["checkpoint_sha256"]
