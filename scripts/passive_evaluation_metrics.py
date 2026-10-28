"""Passive, denominator-explicit metrics for natural complete-match traces.

This module only reads recorded transition snapshots.  It does not select
actions, consume RNG, step the environment, or alter rewards.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any


SCHEMA = "qqt.passive-fullgame-metrics/v2"
PHASES = ("combat_free_enemy", "controlled_window", "carrying", "other")


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return an aggregate ratio, preserving a zero-opportunity denominator."""
    return None if denominator == 0 else float(numerator) / float(denominator)


def rate(numerator: int, denominator: int, *, unit: str) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "unit": unit,
        "value": ratio(numerator, denominator),
    }


def _truth(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def _observed_mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {"value": sum(values) / len(values) if values else None,
            "observed_ticks": len(values), "total_ticks": len(rows)}


def _opportunity_segments(rows, key):
    segments = []
    for row in rows:
        if not _truth(row, key):
            continue
        tick = int(row['tick'])
        if not segments or tick != segments[-1]['end_tick'] + 1:
            segments.append(dict(start_tick=tick, end_tick=tick, ticks=1))
        else:
            segments[-1]['end_tick'] = tick
            segments[-1]['ticks'] += 1
    return segments


def _window_funnel(match_id: str, ticks: list[dict[str, Any]]) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}
    stock_pickup_events: list[dict[str, Any]] = []
    serial = Counter()

    for row in ticks:
        tick = int(row["tick"])
        death_active = (not bool(row.get("enemy_alive_before", True))
                        or int(row.get("enemy_respawn_before", 0) or 0) > 0)
        syrup_active = int(row.get("enemy_trapped_before", 0) or 0) > 0
        active_now = {"death": death_active, "native_syrup": syrup_active}
        starts = {
            "death": _truth(row, "enemy_death"),
            "native_syrup": _truth(row, "enemy_native_trap_enter"),
        }
        # Expiry is determined from the state available for this action. Close
        # before assigning pickups; the first recovered tick is already free.
        for kind in ("death", "native_syrup"):
            if kind in active and not active_now[kind] and not starts[kind]:
                active[kind]["end_tick"] = tick - 1
                del active[kind]
        for kind in ("death", "native_syrup"):
            if starts[kind] and kind not in active:
                serial[kind] += 1
                owner = row.get(f"{kind}_cause_owner", row.get("window_cause_owner"))
                attributed = owner == 0
                window = {
                    "window_id": f"{match_id}:{kind}:{serial[kind]}",
                    "kind": kind,
                    "start_tick": tick,
                    "start_distance_manhattan": row.get("enemy_distance_manhattan"),
                    "cause_owner": owner,
                    "cause_attribution": "self_attributed" if attributed else (
                        "other_attributed" if owner == 1 else "unknown"),
                    "duration_ticks_observed": 0,
                    "stock_pickup_ticks": [],
                    "censored": False,
                    "end_tick": None,
                }
                active[kind] = window
                windows.append(window)
            elif active_now[kind] and kind not in active:
                # Trace can begin inside a window only if an upstream tick is
                # missing.  Keep the event visible but do not invent an onset.
                serial[kind] += 1
                window = {
                    "window_id": f"{match_id}:{kind}:unknown-start:{serial[kind]}",
                    "kind": kind,
                    "start_tick": None,
                    "start_distance_manhattan": None,
                    "cause_owner": None,
                    "cause_attribution": "unknown_start",
                    "duration_ticks_observed": 0,
                    "stock_pickup_ticks": [],
                    "censored": True,
                    "end_tick": None,
                }
                active[kind] = window

            if kind in active and active_now[kind]:
                active[kind]["duration_ticks_observed"] += 1

        if _truth(row, "stock_pickup"):
            candidates = [active[k] for k in ("death", "native_syrup") if k in active]
            if candidates:
                # There can be one controlled opponent; prefer the most recent
                # explicit onset and retain its unique derived window ID.
                candidates.sort(key=lambda w: (w["start_tick"] is not None,
                                                w["start_tick"] or -1), reverse=True)
                candidates[0]["stock_pickup_ticks"].append(tick)
                stock_pickup_events.append({
                    "tick": tick,
                    "classification": "active_control_window",
                    "window_id": candidates[0]["window_id"],
                    "window_kind": candidates[0]["kind"],
                })
            else:
                closed = [w for w in windows if w["end_tick"] is not None
                          and w["start_tick"] is not None and w["end_tick"] < tick]
                closed.sort(key=lambda w: w["end_tick"], reverse=True)
                stock_pickup_events.append({
                    "tick": tick,
                    "classification": "expired_control_window" if closed else "no_observed_control_window",
                    "window_id": closed[0]["window_id"] if closed else None,
                    "window_kind": closed[0]["kind"] if closed else None,
                })


    for window in active.values():
        window["censored"] = True

    for window in windows:
        window['start_to_stock_pickup_ticks'] = (
            [tick - window['start_tick'] for tick in window['stock_pickup_ticks']]
            if window['start_tick'] is not None else None)

    complete = [w for w in windows if not w["censored"] and w["start_tick"] is not None]
    return {
        "windows": windows,
        "stock_pickup_events": stock_pickup_events,
        "by_kind": {
            kind: {
                "window_count": sum(w["kind"] == kind for w in windows),
                "closed_window_count": sum(w["kind"] == kind and not w["censored"] for w in windows),
                "censored_window_count": sum(w["kind"] == kind and w["censored"] for w in windows),
                "windows_with_stock_pickup": sum(
                    w["kind"] == kind and bool(w["stock_pickup_ticks"]) for w in windows),
                "pickup_per_closed_window": rate(
                    sum(w["kind"] == kind and not w["censored"] and bool(w["stock_pickup_ticks"])
                        for w in windows),
                    sum(w["kind"] == kind and not w["censored"] for w in windows),
                    unit="closed_windows"),
                "mean_duration_ticks": (
                    sum(w["duration_ticks_observed"] for w in complete if w["kind"] == kind)
                    / sum(w["kind"] == kind for w in complete)
                    if any(w["kind"] == kind for w in complete) else None),
            }
            for kind in ("death", "native_syrup")
        },
        "reachable_opportunity_proxy": {
            "value": None,
            "reason": "per-tick dynamic wall/brick/pushable path topology was not retained; distance alone is not reachability",
            "denominator": len(complete),
        },
    }


def _carry_episodes(match_id: str, ticks: list[dict[str, Any]], termination: str) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_home_delta: float | None = None
    for row in ticks:
        tick = int(row["tick"])
        before_raw = row.get("own_carried_before", -1)
        after_raw = row.get("own_carried_after", -1)
        before = int(-1 if before_raw is None else before_raw)
        after = int(-1 if after_raw is None else after_raw)
        acquired = before < 0 and after >= 0
        if acquired:
            source = ("enemy_stock" if _truth(row, "stock_pickup") else
                      "loose_enemy" if _truth(row, "loose_enemy_pickup") else
                      "loose_own" if _truth(row, "loose_own_pickup") else
                      "contact_transfer" if _truth(row, "native_transfer") else "unknown")
            current = {
                "carry_episode_id": f"{match_id}:carry:{len(episodes) + 1}",
                "source": source,
                "origin_bun_owner": after,
                "start_tick": tick,
                "end_tick": None,
                "outcome": None,
                "first_return_action_latency_ticks": None,
                "first_movement_action_latency_ticks": None,
                "first_home_distance_regression_tick": None,
                "stalled_ticks": 0,
                "turnback_count": 0,
                "path_length_proxy": 0.0,
                "start_home_distance_manhattan": row.get("home_distance_manhattan_before"),
                "last_home_delta": None,
                "item_placement_ids": [],
            }
            episodes.append(current)
            previous_home_delta = None

        if current is None:
            continue

        if tick > current["start_tick"]:
            if (current["first_movement_action_latency_ticks"] is None
                    and int(row.get("own_move", 4)) != 4):
                current["first_movement_action_latency_ticks"] = tick - current["start_tick"]
            if int(row.get("own_move", 4)) == 4:
                current["stalled_ticks"] += 1
            current["path_length_proxy"] += _number(row, "own_position_delta")
            before_d = row.get("home_distance_manhattan_before")
            after_d = row.get("home_distance_manhattan_after")
            if before_d is not None and after_d is not None:
                delta = float(after_d) - float(before_d)
                if delta < -1e-6 and current['first_return_action_latency_ticks'] is None:
                    current['first_return_action_latency_ticks'] = tick - current['start_tick']
                if (delta > 1e-6 and current["first_home_distance_regression_tick"] is None):
                    current["first_home_distance_regression_tick"] = tick
                if (previous_home_delta is not None and previous_home_delta < -1e-6
                        and delta > 1e-6):
                    current["turnback_count"] += 1
                if abs(delta) > 1e-6:
                    previous_home_delta = delta

        if _truth(row, "item_placed"):
            current["item_placement_ids"].append(f"{match_id}:item:{tick}")

        outcome = None
        if _truth(row, "own_capture"):
            outcome = "delivered"
        elif _truth(row, "own_death"):
            outcome = "death"
        elif _truth(row, "own_drop"):
            outcome = "drop"
        elif int(row.get("own_trapped_after", 0) or 0) > 0 and after < 0:
            outcome = "controlled"
        elif _truth(row, "own_return_bun"):
            outcome = "own_bun_recovered"
        elif _truth(row, "native_transfer") and not acquired and after < 0:
            outcome = "contact_transfer"
        elif after < 0:
            outcome = "returned_or_transferred_unknown"

        if outcome:
            current["end_tick"] = tick
            current["outcome"] = outcome
            current["duration_ticks"] = tick - current["start_tick"] + 1
            start_distance = current["start_home_distance_manhattan"]
            denominator = current["path_length_proxy"]
            current["manhattan_progress_per_path_length_proxy"] = (
                (float(start_distance) / denominator)
                if start_distance is not None and denominator > 0 else None)
            current = None
            previous_home_delta = None

    if current is not None:
        current["outcome"] = "censored_artificial_truncation" if termination == "artificial_truncation" else "censored_terminal"
        current["end_tick"] = int(ticks[-1]["tick"]) if ticks else None
        current["duration_ticks"] = (
            current["end_tick"] - current["start_tick"] + 1
            if current["end_tick"] is not None else None)
        current["manhattan_progress_per_path_length_proxy"] = None
    return episodes


def summarize_game(match_id: str, ticks: list[dict[str, Any]], *,
                   termination: str, winner: int | None,
                   model: str, action_mode: str) -> dict[str, Any]:
    """Aggregate one complete or censored natural match from passive tick rows."""
    rows = [deepcopy(row) for row in ticks]
    rows.sort(key=lambda row: int(row["tick"]))
    if len({int(row["tick"]) for row in rows}) != len(rows):
        raise ValueError("trace tick IDs must be unique within a match")

    total_ticks = len(rows)
    delivery_events = sum(_truth(row, "own_capture") for row in rows)
    death_events = sum(_truth(row, "own_death") for row in rows)
    phase_summary: dict[str, Any] = {}
    for phase in PHASES:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        bomb_opportunities = sum(_truth(row, "bomb_legal") for row in phase_rows)
        release_opportunities = sum(_truth(row, "item_release_opportunity") for row in phase_rows)
        release_requests = sum(_truth(row, "item_release_request") for row in phase_rows)
        phase_summary[phase] = {
            "ticks": len(phase_rows),
            "move_entropy_mean_nats": (
                sum(_number(row, "move_entropy") for row in phase_rows) / len(phase_rows)
                if phase_rows else None),
            "ability_entropy_mean_nats": (
                sum(_number(row, "ability_entropy") for row in phase_rows) / len(phase_rows)
                if phase_rows else None),
            "mean_enemy_distance_manhattan": (
                sum(_number(row, "enemy_distance_manhattan") for row in phase_rows)
                / len(phase_rows) if phase_rows else None),
            "action_probability_means": {
                key: _observed_mean(phase_rows, key)
                for key in ('ability_prob_0', 'ability_prob_1', 'ability_prob_2',
                            'move_prob_0', 'move_prob_1', 'move_prob_2',
                            'move_prob_3', 'move_prob_4')},
            "bomb_placement": {
                "requested_ticks": sum(_truth(row, "bomb_request") for row in phase_rows),
                "legal_opportunity_ticks": bomb_opportunities,
                "placed_events": sum(_truth(row, "bomb_placed") for row in phase_rows),
                "placed_per_request": rate(
                    sum(_truth(row, "bomb_placed") for row in phase_rows),
                    sum(_truth(row, "bomb_request") for row in phase_rows),
                    unit="bomb_request_ticks"),
            },
            "item_release": {
                "inventory_ticks": sum(int(row.get("item_inventory", 0) or 0) > 0 for row in phase_rows),
                "legal_opportunity_ticks": release_opportunities,
                "request_ticks": release_requests,
                "actual_placements": sum(_truth(row, "item_placed") for row in phase_rows),
                "self_attributed_hits": sum(_truth(row, "item_hit_self") for row in phase_rows),
                "enemy_attributed_hits": sum(_truth(row, "item_hit_enemy") for row in phase_rows),
                "release_per_legal_opportunity": rate(
                    sum(_truth(row, "item_placed") for row in phase_rows),
                    release_opportunities, unit="legal_release_ticks"),
                "request_to_placement": rate(
                    sum(_truth(row, "item_placed") for row in phase_rows),
                    release_requests, unit="release_request_ticks"),
                "banana_hit_proxy_ticks": sum(_truth(row, "banana_hit_proxy") for row in phase_rows),
                "glue_hit_proxy_ticks": sum(_truth(row, "glue_hit_proxy") for row in phase_rows),
                "placement_ids": [f"{match_id}:item:{int(row['tick'])}"
                                  for row in phase_rows if _truth(row, "item_placed")],
                "continuous_legal_opportunity_segments": _opportunity_segments(
                    phase_rows, 'item_release_opportunity'),
                "tick_rate_is_utilization_success": False,
            },
        }

    window_funnel = _window_funnel(match_id, rows)
    carry = _carry_episodes(match_id, rows, termination)
    pickups_by_tick = {event["tick"]: event for event in window_funnel["stock_pickup_events"]}
    for episode in carry:
        if episode["source"] == "enemy_stock":
            link = pickups_by_tick.get(episode["start_tick"])
            episode["stock_window_link"] = link if link else {
                "classification": "missing_pickup_event_link",
                "window_id": None,
                "window_kind": None,
            }
    stock_pickups = sum(_truth(row, "stock_pickup") for row in rows)
    free_stock_pickups = sum(_truth(row, "stock_pickup") and row.get("phase") == "combat_free_enemy" for row in rows)
    controlled_stock_pickups = sum(_truth(row, "stock_pickup") and row.get("phase") == "controlled_window" for row in rows)
    loose_enemy_pickups = sum(_truth(row, "loose_enemy_pickup") for row in rows)
    loose_own_recoveries = sum(_truth(row, "loose_own_pickup") for row in rows)
    transfers = sum(_truth(row, "native_transfer") for row in rows)
    carry_deliveries = sum(item["outcome"] == "delivered" for item in carry)
    closed_carry = [item for item in carry if item["outcome"] not in {
        "censored_artificial_truncation", "censored_terminal"}]
    legacy_returns = [_number(row, "legacy_reward") for row in rows]
    minimal_returns = [_number(row, "minimal_reward") for row in rows]

    return {
        "schema": SCHEMA,
        "match_id": match_id,
        "model": model,
        "action_mode": action_mode,
        "termination": termination,
        "winner_p0_p1_draw_unknown": winner,
        "valid_ticks": total_ticks,
        "full_game": {
            "game_count": 1,
            "at_least_one_delivery": rate(int(delivery_events > 0), 1, unit="natural_matches"),
            "delivery_events": {"count": int(delivery_events), "denominator": "match"},
            "own_deaths": {"count": int(death_events), "denominator": "match"},
            "net_wins": rate(int(winner == 0), 1, unit="natural_matches") if winner is not None else None,
            "net_losses": rate(int(winner == 1), 1, unit="natural_matches") if winner is not None else None,
            "draw_or_unknown": winner in (None, -1, -2),
            "termination_class": termination,
        },
        "phases": phase_summary,
        "window_funnel": window_funnel,
        "pickup_sources": {
            "enemy_stock": {"count": int(stock_pickups), "denominator": "match"},
            "enemy_stock_during_controlled_window": {"count": int(controlled_stock_pickups), "denominator": "match"},
            "enemy_stock_while_enemy_free": {"count": int(free_stock_pickups), "denominator": "match"},
            "loose_enemy_bun": {"count": int(loose_enemy_pickups), "denominator": "match"},
            "own_bun_recovery": {"count": int(loose_own_recoveries), "denominator": "match"},
            "native_transfer": {"count": int(transfers), "denominator": "match"},
        },
        "carry_episodes": {
            "episodes": carry,
            "count": len(carry),
            "same_episode_delivery_rate": rate(carry_deliveries, len(closed_carry), unit="closed_carry_episodes"),
            "outcomes": dict(Counter(item["outcome"] for item in carry)),
            "first_return_action_latency_ticks": [item["first_return_action_latency_ticks"] for item in carry],
            "path_measure": "Manhattan home-distance regression and observed movement-length proxies; not a dynamic route oracle",
        },
        "reward_alignment": {
            "undiscounted_legacy_return": float(sum(legacy_returns)),
            "undiscounted_minimal_return": float(sum(minimal_returns)),
            "rewards_are_cross_scale_comparable": False,
            "exclusive_decomposition_checked_by_fullmatch_runner": True,
        },
        "coverage": {
            "match_outcome": "observed",
            "phase_action_entropy": "observed_per_tick",
            "window_ids": "derived_unique_ids_from_native_death_or_trap_edges; not source-engine event IDs",
            "dynamic_reachability": "missing",
            "oracle_first_deviation": "unavailable_in_this_native_fullmatch_trace",
            "complete_item_net_benefit": "not_estimated; observed effect completion is a proxy, not a causal net value",
        },
    }


def summarize_cohort(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool counts before division; never average per-match rates."""
    n = len(games)
    delivery_matches = sum(g["full_game"]["at_least_one_delivery"]["numerator"] for g in games)
    death_matches = sum(g["full_game"]["own_deaths"]["count"] > 0 for g in games)
    deliveries = sum(g["full_game"]["delivery_events"]["count"] for g in games)
    deaths = sum(g["full_game"]["own_deaths"]["count"] for g in games)
    return {
        "games": n,
        "at_least_one_delivery_rate": rate(delivery_matches, n, unit="natural_matches"),
        "any_own_death_rate": rate(death_matches, n, unit="natural_matches"),
        "delivery_events_per_match": rate(deliveries, n, unit="natural_matches"),
        "own_deaths_per_match": rate(deaths, n, unit="natural_matches"),
        "net_wins": rate(sum(g["full_game"]["net_wins"]["numerator"]
                             for g in games if g["full_game"]["net_wins"] is not None),
                         sum(1 for g in games if g["full_game"]["net_wins"] is not None),
                         unit="natural_matches"),
        "termination_counts": dict(Counter(g["termination"] for g in games)),
        "pickup_event_counts": {
            key: sum(g["pickup_sources"][key]["count"] for g in games)
            for key in games[0]["pickup_sources"]
        } if games else {},
        "phase_ticks": {
            phase: sum(g["phases"][phase]["ticks"] for g in games)
            for phase in PHASES
        },
        "windows": {
            kind: _pooled_windows(games, kind) for kind in ('death', 'native_syrup')},
        "carry_episodes": _pooled_carry(games),
        "phases": {phase: _pooled_phase(games, phase) for phase in PHASES},
    }


def _pooled_windows(games, kind):
    windows = [w for g in games for w in g['window_funnel']['windows'] if w['kind'] == kind]
    closed = [w for w in windows if not w['censored'] and w['start_tick'] is not None]
    return dict(
        window_count=len(windows), closed_window_count=len(closed),
        censored_window_count=sum(w['censored'] for w in windows),
        pickup_per_closed_window=rate(sum(bool(w['stock_pickup_ticks']) for w in closed),
                                     len(closed), unit='closed_windows'),
        cause_attribution_counts=dict(Counter(w['cause_attribution'] for w in windows)),
        observed_duration_ticks=[w['duration_ticks_observed'] for w in windows],
        start_to_pickup_ticks=[t-w['start_tick'] for w in windows
                              if w['start_tick'] is not None for t in w['stock_pickup_ticks']])


def _pooled_carry(games):
    episodes = [e for g in games for e in g['carry_episodes']['episodes']]
    closed = [e for e in episodes if not e['outcome'].startswith('censored_')]
    return dict(count=len(episodes), closed_count=len(closed),
                censored_count=len(episodes)-len(closed),
                outcomes=dict(Counter(e['outcome'] for e in episodes)),
                source_counts=dict(Counter(e['source'] for e in episodes)),
                same_episode_delivery_rate=rate(sum(e['outcome']=='delivered' for e in closed),
                                               len(closed), unit='closed_carry_episodes'))


def _pooled_phase(games, phase):
    phases = [g['phases'][phase] for g in games]
    ticks = sum(p['ticks'] for p in phases)
    # Entropies are tick means, while opportunities and events pool raw counts.
    result = dict(ticks=ticks)
    for key in ('move_entropy_mean_nats', 'ability_entropy_mean_nats',
                'mean_enemy_distance_manhattan'):
        observed = sum(p['ticks'] for p in phases if p.get(key) is not None)
        result[key] = (sum(p[key]*p['ticks'] for p in phases if p.get(key) is not None)
                       / observed if observed else None)
    for group, keys in (
            ('bomb_placement', ('requested_ticks','legal_opportunity_ticks','placed_events')),
            ('item_release', ('inventory_ticks','legal_opportunity_ticks','request_ticks',
                              'actual_placements','self_attributed_hits','enemy_attributed_hits',
                              'banana_hit_proxy_ticks','glue_hit_proxy_ticks'))):
        result[group] = {key: sum(p[group][key] for p in phases) for key in keys}
    result['item_release']['request_to_placement'] = rate(
        result['item_release']['actual_placements'], result['item_release']['request_ticks'],
        unit='release_request_ticks')
    result['item_release']['legal_opportunity_segments'] = sum(
        len(p['item_release'].get('continuous_legal_opportunity_segments', [])) for p in phases)
    result['action_probability_means'] = {}
    for key in ('ability_prob_0','ability_prob_1','ability_prob_2','move_prob_4'):
        fields = [p.get('action_probability_means', {}).get(key, {}) for p in phases]
        observed = sum(f.get('observed_ticks', 0) for f in fields)
        result['action_probability_means'][key] = dict(
            observed_ticks=observed, total_ticks=ticks,
            value=sum(f['value']*f['observed_ticks'] for f in fields if f.get('value') is not None)
                  / observed if observed else None)
    return result


def coverage_matrix() -> dict[str, Any]:
    """Static coverage states for the current per-tick native full-game schema."""
    return {
        "implemented": {
            "full_game_outcomes": ["delivery match rate", "delivery event count", "own deaths", "wins/losses/draw/timeout/censor"],
            "phase_exposure": ["combat/free-enemy ticks", "controlled-window ticks", "carrying ticks", "distance proxy", "move/ability entropy", "bomb/item request and legality"],
            "window_chain": ["derived unique death/trap window IDs", "observed duration", "window-associated stock pickups", "death/native-syrup separated"],
            "carry_chain": ["source", "delivery/drop/death/control/censor", "first return action", "stall", "Manhattan regression", "movement length proxy"],
            "items": ["inventory ticks", "legal opportunity ticks", "release requests/placements", "self/enemy hit attribution", "banana slide/glue slow hit proxies; not full chain completion"],
            "reward": ["native legacy/minimal per-transition decomposition", "discounted/undiscounted returns", "ledger snapshots"],
        },
        "recomputable_from_existing_logs": {
            "training_optimizer": ["approx_kl", "clip_fraction", "entropy", "loss", "update_l2", "ability samples", "item opportunity counts", "env/optimizer steps", "throughput", "actual opponent exposure"],
            "source": "formal/*/updates.jsonl and committed checkpoint metrics; these are training-batch aggregates, not full-game policy metrics",
        },
        "missing_data": {
            "legacy_eval_backfill": "Existing CPU/JS evaluator summaries do not retain per-tick states/actions/window IDs; cannot reconstruct this schema without rerunning those checkpoint evaluations.",
            "critic_ev": "No per-tick V was exported; value loss and GAE are not EV. Requires value predictions and declared full/bootstrapped targets, with zero variance undefined.",
            "common_state_policy_kl": "No frozen common-state probe rows have been serialized by the existing evaluator; own-occupancy entropy is not a substitute.",
            "dynamic_route_reachability": "No obstacle grids/path query is retained in current compact trace; Manhattan proximity is not reachability.",
            "exact_failure_cause": "Only directly observed terminal events are classified; unlinked pursuit/route deviation is unknown.",
            "placement_to_control_kill_chain": "No unique bomb placement-to-control/kill ID witness is retained. Phase bomb counts are observed; attribution of a later control to a particular placement is unknown.",
            "danger_stratification_and_self_trade": "Selected-action danger, self-harm and simultaneous-death flags have not been serialized in this compact trace; no inferred zeros.",
            "interaction_durations_and_distance_distribution": "Mean Manhattan distance is retained; no declared contact/interaction predicate or binned distance statistic in prior summaries.",
            "complete_item_chain": "Placement tick IDs do not link to consumed effect IDs. Self/enemy contact owner is observed, but full slide/preparation/risk/old-overlap/censor/benefit per placement is unknown.",
            "item_kind_exposure": "Inventory is a boolean in this trace; banana/glue-specific inventory/opportunity denominators are unknown.",
            "intermediate_reward_ledgers": "Per-transition reward terms and final ledger snapshots are retained. Intermediate pending/payment/clawback balances are not complete ledger snapshots.",
            "initial_game_geometry": "New trace batches include a wall/brick fingerprint, not a human map catalog identity or complete geometry replay.",
        },
        "unavailable": {
            "production_js_hunter": "This evaluation is native JAX against fixed JAX rule3; no JS Hunter result is inferred.",
            "causal_item_net_benefit": "No matched no-item counterfactual; placement/hit/completion observations are not causal value.",
            "oracle_first_deviation": "This full-game runner has no serialized route oracle/action-set witness; a distance regression is not oracle deviation.",
            "training_seed_generalization": "The first pair uses one parent training seed and one candidate training seed; game-seed pairing does not estimate training-seed variance.",
        },
    }
