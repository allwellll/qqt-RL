import json
import pickle

import numpy as np

from qqt_rl.training.config import family_for_cycle, validate_cycle_count
from qqt_rl.training.io import atomic_copy, atomic_write_json, checkpoint_is_finite
from scripts.prepare_bun_safe_aggression_v7 import build_common_config
from scripts.launch_bun_safe_aggression_v7 import process_matches, require_under


def test_family_schedule_keeps_frozen_r6_semantics():
    config = {"family_curriculum": True, "total_cycles": 360}
    assert family_for_cycle(config, 1)["name"] == "easy"
    assert family_for_cycle(config, 5)["name"] == "full_v2"
    assert family_for_cycle(config, 200)["name"] == "full_v2"
    assert family_for_cycle(config, 360)["name"] == "full_v2"
    assert validate_cycle_count(1) == 1
    assert validate_cycle_count(360) == 360


def test_prepare_enables_repaired_v7_seed_namespace():
    config = build_common_config(360, 90)
    assert config["critic_seed_namespace_version"] == "v7"
    assert config["total_cycles"] == 360
    assert config["persistent_worker"] is True
    assert config["counterfactual_engine"] == "batched"
    assert config["counterfactual_state_batch_size"] == 4
    assert config["critic_batch_size"] == 64
    assert config["jax_cache_min_compile_time_secs"] == 0.0
    assert config["opponent_bot"]["spec"]["id"] == "bun.tactical_v2"


def test_manifest_paths_cannot_escape_repository(tmp_path):
    assert require_under(tmp_path / "runs/demo", tmp_path, "run") == tmp_path / "runs/demo"
    try:
        require_under(tmp_path / "../outside", tmp_path, "run")
    except ValueError as error:
        assert "repository root" in str(error)
    else:
        raise AssertionError("escaped path accepted")


def test_worker_identity_rejects_unrelated_live_pid(tmp_path):
    record = {"pid": __import__("os").getpid(), "start_ticks": 0}
    assert not process_matches(record, tmp_path / "config.json")


def test_atomic_artifact_helpers(tmp_path):
    manifest = tmp_path / "status.json"
    atomic_write_json(manifest, {"cycle": 9})
    assert json.loads(manifest.read_text()) == {"cycle": 9}
    assert not manifest.with_suffix(".json.tmp").exists()

    source = tmp_path / "source.pkl"
    destination = tmp_path / "published.pkl"
    source.write_bytes(pickle.dumps({"params": {"w": np.ones((2, 2))}}))
    atomic_copy(source, destination)
    assert checkpoint_is_finite(destination)
    assert source.read_bytes() == destination.read_bytes()
