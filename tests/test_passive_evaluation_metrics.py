from copy import deepcopy

from scripts import passive_evaluation_metrics as metrics


def tick(t, **kwargs):
    row = {
        "tick": t,
        "phase": "combat_free_enemy",
        "enemy_alive_before": True,
        "enemy_alive_after": True,
        "enemy_trapped_before": 0,
        "enemy_respawn_before": 0,
        "enemy_death": False,
        "enemy_native_trap_enter": False,
        "window_cause_owner": -1,
        "enemy_distance_manhattan": 7,
        "own_carried_before": -1,
        "own_carried_after": -1,
        "own_capture": False,
        "own_return_bun": False,
        "own_death": False,
        "own_drop": False,
        "own_trapped_after": 0,
        "stock_pickup": False,
        "loose_enemy_pickup": False,
        "loose_own_pickup": False,
        "native_transfer": False,
        "own_move": 4,
        "own_ability": 0,
        "move_entropy": 1.0,
        "ability_entropy": 0.5,
        "bomb_legal": False,
        "bomb_request": False,
        "bomb_placed": False,
        "item_inventory": 0,
        "item_release_opportunity": False,
        "item_release_request": False,
        "item_placed": False,
        "item_hit_self": False,
        "item_hit_enemy": False,
        "banana_completed": False,
        "glue_completed": False,
        "own_position_delta": 0.0,
        "home_distance_manhattan_before": 8.0,
        "home_distance_manhattan_after": 8.0,
        "legacy_reward": 0.0,
        "minimal_reward": 0.0,
    }
    row.update(kwargs)
    return row


def test_zero_opportunity_remains_undefined_not_zero():
    result = metrics.rate(0, 0, unit="legal_release_ticks")
    assert result == {
        "numerator": 0,
        "denominator": 0,
        "unit": "legal_release_ticks",
        "value": None,
    }


def test_death_and_native_syrup_windows_have_unique_ids_and_separate_denominators():
    rows = [
        tick(0),
        tick(1, enemy_death=True, window_cause_owner=0),
        tick(2, enemy_alive_before=False, enemy_alive_after=False,
             enemy_respawn_before=5, stock_pickup=True, phase="controlled_window"),
        tick(3, enemy_alive_before=False, enemy_alive_after=False,
             enemy_respawn_before=4, phase="controlled_window"),
        tick(4, enemy_native_trap_enter=True, window_cause_owner=1,
             stock_pickup=True, phase="controlled_window"),
        tick(5, enemy_trapped_before=3, stock_pickup=True,
             phase="controlled_window"),
        tick(6),
    ]
    summary = metrics.summarize_game(
        "m1", rows, termination="game_timeout", winner=0,
        model="parent", action_mode="categorical")
    windows = summary["window_funnel"]["windows"]
    assert [w["kind"] for w in windows] == ["death", "native_syrup"]
    assert len({w["window_id"] for w in windows}) == 2
    assert windows[0]["stock_pickup_ticks"] == [2]
    assert windows[1]["stock_pickup_ticks"] == [4, 5]
    assert windows[0]["duration_ticks_observed"] == 2
    assert windows[1]["duration_ticks_observed"] == 1
    assert summary["window_funnel"]["by_kind"]["death"]["pickup_per_closed_window"] == {
        "numerator": 1, "denominator": 1, "unit": "closed_windows", "value": 1.0}


def test_multiple_carry_outcomes_keep_closed_and_censored_denominators_separate():
    rows = [
        tick(0, own_carried_after=1, stock_pickup=True),
        tick(1, own_carried_before=1, own_carried_after=1,
             own_move=0, home_distance_manhattan_before=8,
             home_distance_manhattan_after=9, own_position_delta=1),
        tick(2, own_carried_before=1, own_death=True),
        tick(3, own_carried_after=1, loose_enemy_pickup=True),
        tick(4, own_carried_before=1, own_drop=True),
        tick(5, own_carried_after=1, stock_pickup=True),
        tick(6, own_carried_before=1, own_carried_after=1, own_move=3),
    ]
    summary = metrics.summarize_game(
        "m2", rows, termination="artificial_truncation", winner=None,
        model="candidate", action_mode="greedy")
    carries = summary["carry_episodes"]
    assert [x["outcome"] for x in carries["episodes"]] == [
        "death", "drop", "censored_artificial_truncation"]
    assert carries["same_episode_delivery_rate"] == {
        "numerator": 0, "denominator": 2,
        "unit": "closed_carry_episodes", "value": 0.0}
    assert carries["episodes"][0]["first_home_distance_regression_tick"] == 1
    assert carries["episodes"][2]["duration_ticks"] == 2


def test_item_placement_ids_are_event_scoped_and_phase_counts_use_tick_denominators():
    rows = [
        tick(0, phase="carrying", item_inventory=1,
             item_release_opportunity=True, item_release_request=True,
             item_placed=True),
        tick(1, phase="carrying", item_inventory=1,
             item_release_opportunity=True, item_release_request=True,
             item_placed=True, item_hit_enemy=True, banana_hit_proxy=True),
    ]
    summary = metrics.summarize_game(
        "m3", rows, termination="native_game_end", winner=1,
        model="candidate", action_mode="categorical")
    item = summary["phases"]["carrying"]["item_release"]
    assert item["release_per_legal_opportunity"] == {
        "numerator": 2, "denominator": 2, "unit": "legal_release_ticks", "value": 1.0}
    assert item["request_to_placement"]["denominator"] == 2
    assert len(item["placement_ids"]) == len(set(item["placement_ids"])) == 2
    assert item["enemy_attributed_hits"] == 1
    assert item["banana_hit_proxy_ticks"] == 1
    assert summary["full_game"]["net_losses"] == {
        "numerator": 1, "denominator": 1, "unit": "natural_matches", "value": 1.0}


def test_summary_does_not_mutate_observed_trace_and_reward_scales_stay_separate():
    rows = [tick(0, legacy_reward=3.0, minimal_reward=-2.0)]
    original = deepcopy(rows)
    summary = metrics.summarize_game(
        "m4", rows, termination="native_game_end", winner=-1,
        model="candidate", action_mode="greedy")
    assert rows == original
    assert summary["reward_alignment"]["undiscounted_legacy_return"] == 3.0
    assert summary["reward_alignment"]["undiscounted_minimal_return"] == -2.0
    assert summary["reward_alignment"]["rewards_are_cross_scale_comparable"] is False
    assert summary["window_funnel"]["reachable_opportunity_proxy"]["value"] is None


def test_duplicate_tick_rejects_double_counted_window_or_item_events():
    rows = [tick(0, item_placed=True), tick(0, item_placed=True)]
    try:
        metrics.summarize_game("m5", rows, termination="native_game_end",
                               winner=None, model="candidate", action_mode="greedy")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate event ticks must not inflate denominators")


def test_recovered_enemy_closes_window_before_a_free_alive_pickup():
    rows = [tick(0, enemy_death=True),
            tick(1, enemy_alive_before=False, enemy_respawn_before=1),
            tick(2, stock_pickup=True)]
    result = metrics.summarize_game('recovered', rows, termination='game_timeout',
                                    winner=-1, model='p', action_mode='greedy')
    event = result['window_funnel']['stock_pickup_events'][0]
    assert event['classification'] == 'expired_control_window'
    assert result['window_funnel']['windows'][0]['stock_pickup_ticks'] == []


def test_contact_transfer_acquisition_remains_open_until_delivery_after_respawn():
    rows = [tick(0, native_transfer=True, own_carried_after=1),
            tick(1, own_carried_before=1, own_carried_after=1,
                 enemy_alive_before=True, own_move=3),
            tick(2, own_carried_before=1, own_capture=True)]
    result = metrics.summarize_game('transfer', rows, termination='native_capture',
                                    winner=0, model='p', action_mode='categorical')
    episodes = result['carry_episodes']['episodes']
    assert len(episodes) == 1
    assert episodes[0]['source'] == 'contact_transfer'
    assert episodes[0]['outcome'] == 'delivered'
    assert episodes[0]['end_tick'] == 2


def test_cohort_window_rates_pool_counts_and_keep_censored_separate():
    def game(mid, rows):
        return metrics.summarize_game(mid, rows, termination='game_timeout',
                                      winner=-1, model='p', action_mode='greedy')
    a = game('a', [tick(0, enemy_death=True),
                   tick(1, enemy_alive_before=False, stock_pickup=True), tick(2)])
    b = game('b', [tick(0, enemy_death=True), tick(1, enemy_alive_before=False),
                   tick(2), tick(3, enemy_death=True),
                   tick(4, enemy_alive_before=False), tick(5),
                   tick(6, enemy_death=True)])
    result = metrics.summarize_cohort([a, b])
    assert result['windows']['death']['pickup_per_closed_window'] == {
        'numerator': 1, 'denominator': 3, 'unit': 'closed_windows', 'value': 1/3}
    assert result['windows']['death']['censored_window_count'] == 1


def test_passive_trace_collector_contains_no_rng_or_environment_step():
    from pathlib import Path
    source = Path("scripts/expert_reward_alignment_fullmatch.py").read_text()
    collector = source.split("def _passive_snapshot", 1)[1].split("\ndef source_model_rows", 1)[0]
    assert "jax.random" not in collector
    assert ".step(" not in collector
    names = __import__("scripts.expert_reward_alignment_fullmatch", fromlist=["TELEMETRY_NAMES"]).TELEMETRY_NAMES
    assert len(names) == len(set(names))
    assert {'own_capture', 'winner', 'death_cause_owner', 'native_syrup_cause_owner'} <= set(names)
