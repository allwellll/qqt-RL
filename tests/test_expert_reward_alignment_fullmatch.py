from tests.external_artifacts import requires_artifacts
import json
from pathlib import Path

import pytest

from scripts import expert_reward_alignment_fullmatch as fullmatch


def test_complete_match_budget_uses_only_exact_scan_chunks():
    sizes = fullmatch.chunk_sizes()
    assert sizes == [128] * 20
    assert sum(sizes) == 2560
    assert fullmatch.MAX_TICKS >= 2400 + 160


def test_chunk_scheduler_rejects_a_silent_partial_match_tail():
    with pytest.raises(ValueError, match="positive multiple"):
        fullmatch.chunk_sizes(total=2600, chunk=128)


def test_manifest_scope_covers_each_paired_source_and_both_action_modes():
    assert set(fullmatch.MODELS) >= {
        "parent_r1_s2751_u064",
        "parent_r1_s2752_u064",
        "parent_r1_s2753_u064",
        "legacy_s2851_u320",
        "minimal_s2851_u064",
        "r2_early_stopped_s2751_u1664",
    }
    assert fullmatch.GAMES_PER_MODE == 32
    assert len(fullmatch.MODELS) * 2 * fullmatch.GAMES_PER_MODE == 384
    assert fullmatch.OUT.name == "fullmatch_reward_alignment_v2"


@requires_artifacts('runs/expert_reward_20261025/D_ablation_v1/PROTOCOL.json')
def test_d_reward_protocol_and_frozen_coefficients_are_not_rewritten():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "runs/expert_reward_20261025/D_ablation_v1/PROTOCOL.json").read_text()
    )
    adapter = (root / "scripts/expert_reward_ablation.py").read_text()
    assert protocol["reward_package"]["minimal"]["delivery"] == "+2.0 per native capture event"
    assert "DELIVERY_REWARD = 2.0" in adapter
    assert fullmatch.OUT.name == "fullmatch_reward_alignment_v2"


def test_exclusive_component_partition_recomposes_both_reward_streams():
    names = (*fullmatch.EXCLUSIVE_LEGACY_TERMS, "combined_total")
    values = list(range(1, len(fullmatch.EXCLUSIVE_LEGACY_TERMS) + 1))
    total = sum(values)
    parts = __import__("numpy").array([[values + [total]]], dtype=float)
    reward = __import__("numpy").array([[total]], dtype=float)
    result = fullmatch.per_transition_decomposition_error(
        parts, reward, names, fullmatch.EXCLUSIVE_LEGACY_TERMS)
    assert result["max_abs_exclusive_to_total"] == 0.0
    assert result["max_abs_total_component_to_reward"] == 0.0
    assert "combined_total" not in fullmatch.EXCLUSIVE_LEGACY_TERMS
    assert "total" not in fullmatch.EXCLUSIVE_LEGACY_TERMS


def test_potential_clawback_is_included_without_double_counting_phase_metrics():
    names = tuple(dict.fromkeys((
        *fullmatch.EXCLUSIVE_LEGACY_TERMS, "potential_clawback",
        "drop_approach_clawback", "carry_home_clawback", "total",
        "combined_total",
    )))
    values = {name: 0.0 for name in names}
    values.update(global_safety=-12.5, clawback=-1.9,
                  transport_clawback=-0.75, potential_clawback=-0.25,
                  drop_approach_clawback=-0.25,
                  carry_home_clawback=-0.25, total=-14.4,
                  combined_total=-15.4)
    parts = __import__("numpy").array(
        [[[values[name] for name in names]]], dtype=float)
    reward = __import__("numpy").array([[-15.4]], dtype=float)

    result = fullmatch.per_transition_decomposition_error(
        parts, reward, names, fullmatch.EXCLUSIVE_LEGACY_TERMS)

    assert result["max_abs_exclusive_to_total"] < 1e-8
    assert result["max_abs_total_component_to_reward"] == 0.0
    assert "potential_clawback" in fullmatch.EXCLUSIVE_LEGACY_TERMS
    assert "drop_approach_clawback" not in fullmatch.EXCLUSIVE_LEGACY_TERMS
    assert "carry_home_clawback" not in fullmatch.EXCLUSIVE_LEGACY_TERMS


def test_global_progress_sums_games_rather_than_counting_condition_rows():
    rows = [{'games': 32}, {'games': 32}]
    assert fullmatch.completed_game_count(rows, 8) == 72


def test_native_winner_uses_first_terminal_event_not_settlement_tail():
    rows = [{'winner': -2}, {'winner': 0}, {'winner': -2}, {'winner': 1}]
    assert fullmatch.native_winner(rows) == 0
    assert fullmatch.native_winner([{'winner': -2}]) is None


def test_numeric_trace_artifact_roundtrips_exact_values_and_column_names(tmp_path):
    import numpy as np
    arrays = tuple(np.arange(16, dtype=np.float32).reshape(8, 2) for _ in range(9))
    path = fullmatch.write_batch_trace(tmp_path, 'parent', 'greedy', [0,1],
                                      arrays, ('a','combined_total'))
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['legacy_reward'], arrays[0])
        np.testing.assert_array_equal(saved['telemetry'], arrays[8])
        assert saved['component_names'].tolist() == ['a', 'combined_total']
        assert saved['episode_indices'].tolist() == [0,1]


def test_passive_snapshot_reads_immutable_state_actions_and_leaves_rng_untouched(monkeypatch):
    """Execute the production reader on read-only numeric state, without JIT."""
    from types import SimpleNamespace
    import pickle
    import random
    import numpy as np

    monkeypatch.setattr(fullmatch, 'jnp', np)
    monkeypatch.setattr(fullmatch.E, 'env', SimpleNamespace(
        _BUN_BASES=np.array([[1,1],[9,11]]), STATUS_SLIDE=1, STATUS_SLOW=2))
    grid = lambda: np.zeros((2,13,15), np.int32)
    players = lambda: np.zeros((2,2), np.int32)
    state = SimpleNamespace(t=np.zeros(2, np.int32),
        pos=np.full((2,2,2), 3.5), core=SimpleNamespace(blast_linger=grid()),
        field_item=grid(), held_item=players(), wall=grid(), brick=grid(), fuse=grid(),
        alive=np.ones((2,2), bool), trapped=players(), bun_carried=players()-1,
        trapped_causal=np.zeros((2,2,2),bool), trapped_source=np.zeros((2,2,2),bool),
        move_status=players(), bun_respawn=players())
    info = {name:players() for name in (
        'trap_hit','trap_owner','causal_kill','native_trap_popper','own_bomb_defeat',
        'death','native_trap_enter','capture','return_bun','drop','steal',
        'pickup_loose','pickup_origin','native_transfer','bomb_placed')}
    info.update(causal_death_source=np.zeros((2,2,2),bool), winner=np.full(2,-2))
    args = (state,state,info,np.zeros((2,2,2),np.int32),
            (np.full((2,5),.2),np.full((2,3),1/3)),
            np.ones((2,5),bool),np.ones((2,3),bool),
            np.zeros((2,2)),np.zeros((2,2)),np.ones(2,bool),np.zeros(2,bool))

    def readonly(value):
        if isinstance(value,np.ndarray):
            value.setflags(write=False)
        elif isinstance(value,SimpleNamespace):
            readonly(vars(value))
        elif isinstance(value,dict):
            for v in value.values(): readonly(v)
        elif isinstance(value,(tuple,list)):
            for v in value: readonly(v)
    readonly(args)
    before = pickle.dumps(args)
    rng_before = pickle.dumps((np.random.get_state(),random.getstate()))
    first = fullmatch._passive_snapshot(*args)
    second = fullmatch._passive_snapshot(*args)
    np.testing.assert_array_equal(first,second)
    assert first.shape == (2,len(fullmatch.TELEMETRY_NAMES))
    assert pickle.dumps(args) == before
    assert pickle.dumps((np.random.get_state(),random.getstate())) == rng_before
