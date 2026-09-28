import hashlib

import numpy as np

from jax_bomb.bun_expert import (
    PHASE_KILL_COMBAT,
    PHASE_KILL_RETURN,
    PHASE_KILL_RUSH,
    PHASE_NAMES,
    rollout_expert,
)


def _action_digest(result):
    actions = np.asarray([
        [frame["move_action"], frame["ability_action"]]
        for frame in result["frames"]
    ], np.uint8)
    return hashlib.sha256(actions.tobytes()).hexdigest()


def test_rule_expert_is_deterministic_and_completes_full_bun_loop():
    first = rollout_expert(202609250000)
    second = rollout_expert(202609250000)
    for result in (first, second):
        assert result["success"]
        assert result["owner_walls"] > 0
        assert result["own_detonations"] > 0
        assert result["safe_detonations"] == result["own_detonations"]
        assert result["entered_enemy_base"]
        assert result["stole"]
        assert result["crossed_home"]
        assert result["captured"]
        assert result["failure"] is None
        assert result["frames"]
        for frame in result["frames"]:
            assert frame["move_mask"][frame["move_action"]]
            assert frame["ability_mask"][frame["ability_action"]]
            assert int(frame["phase"]) < len(PHASE_NAMES)
    assert first["length"] == second["length"]
    assert _action_digest(first) == _action_digest(second)


def test_rule_expert_solves_route_and_objective_rehearsal_lessons():
    contracts = {
        "route_break=1": "route_success",
        "route_to_base=1": "bridge_success",
        "near_steal=1": "stole",
        "carry_home=1": "captured",
        "carry_return=1": "captured",
        "combat_static=1": "credited_hit",
        "combat_moving=1": "credited_hit",
        "combat_kill=1": "causal_kill",
        "kill_rush=1": "captured",
    }
    for offset, (curriculum, required) in enumerate(contracts.items()):
        result = rollout_expert(
            202609258000 + offset, record=False, curriculum=curriculum)
        assert result["success"], (curriculum, result["failure"])
        assert result[required]
        assert not result["self_death"]
        assert result["safe_detonations"] == result["own_detonations"]


def test_kill_rush_expert_uses_respawn_window_and_phase_sequence():
    first = rollout_expert(
        202609256000, record=True, curriculum="kill_rush=1")
    second = rollout_expert(
        202609256000, record=True, curriculum="kill_rush=1")

    assert first["success"] and second["success"]
    assert first["causal_kill"] and first["credited_kill"]
    assert not first["trade"]
    assert not first["own_bomb_defeat"]
    assert first["first_kill_tick"] == second["first_kill_tick"]
    assert first["first_steal_tick"] == second["first_steal_tick"]
    assert 0 < first["steal_respawn_remaining"] <= 100
    assert first["first_kill_tick"] < first["first_enemy_base_tick"]
    assert first["first_enemy_base_tick"] <= first["first_steal_tick"]
    assert first["first_steal_tick"] < first["first_cross_tick"]
    assert first["first_cross_tick"] < first["first_capture_tick"]
    phases = {int(frame["phase"]) for frame in first["frames"]}
    assert PHASE_KILL_COMBAT in phases
    assert PHASE_KILL_RUSH in phases
    assert PHASE_KILL_RETURN in phases
    assert _action_digest(first) == _action_digest(second)


def test_full_ambush_expert_produces_safe_surviving_kill():
    result = rollout_expert(
        202609260700, record=True, curriculum="full_ambush=1")
    assert result["success"], result["failure"]
    assert result["causal_kill"]
    assert result["credited_kill"]
    assert not result["trade"]
    assert not result["own_bomb_defeat"]
    assert not result["self_death"]
    assert result["safe_detonations"] == result["own_detonations"]


def test_full_ambush_expert_recovers_from_safe_counterfactual_actions():
    first = rollout_expert(
        202609260701, record=True, curriculum="full_ambush=1",
        recovery_perturbations=2)
    second = rollout_expert(
        202609260701, record=True, curriculum="full_ambush=1",
        recovery_perturbations=2)
    assert first["success"], first["failure"]
    assert first["recovery_perturbations"] == 2
    assert first["safe_detonations"] == first["own_detonations"]
    assert _action_digest(first) == _action_digest(second)
