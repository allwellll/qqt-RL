"""Read-only training-side aggregation for bomb-threat runs.

The collector writes per-update counters.  This module only aggregates those
counters and records unavailable fields explicitly; it never changes training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs" / "bomb_threat_v1"
OPPONENTS = (
    "hunter_tier3a", "hunter_tier3b", "jax_hunter_hard", "selfplay_current",
    "selfplay_ring", "selfplay_fixed",
)

# The first 82 entries are the production native reward schema.  The collector
# replaces its old combined-total slot with six item diagnostics and appends a
# new combined total, yielding 90 entries.
NATIVE_COMPONENTS_82 = (
    "lesson_task", "global_safety", "settled_kill", "settled_shaping",
    "clawback", "blocked_route", "total", "transport_shape",
    "transport_clawback", "transport_capture", "drop_bonus", "drop_clawback",
    "drop_approach_positive", "drop_approach_negative", "drop_approach_net",
    "carry_home_positive", "carry_home_negative", "carry_home_net",
    "potential_stationary", "potential_phase_start", "potential_phase_end",
    "potential_cap_hit", "potential_topology_rebase", "potential_clawback",
    "drop_approach_stationary", "drop_approach_phase_start",
    "drop_approach_phase_end", "drop_approach_cap_hit",
    "drop_approach_topology_rebase", "drop_approach_clawback",
    "carry_home_stationary", "carry_home_phase_start", "carry_home_phase_end",
    "carry_home_cap_hit", "carry_home_topology_rebase", "carry_home_clawback",
    "bridge_reward", "bridge_clawback", "bridge_drop", "bridge_pickup",
    "bridge_switch", "fixture_kill_suppressed", "glue_placed",
    "glue_qualified_placement", "glue_attributed_hit", "banana_start",
    "banana_end", "banana_before", "banana_after", "banana_gain",
    "item_positive", "item_clawback", "item_capture", "item_alive_carry",
    "item_net", "h2_raw_custody_ticks", "h2_custody_item_available_ticks",
    "h2_custody_glue_free_enemy_ticks", "h2_empty_combat_ticks",
    "h2_banana_inflight_ticks", "control_entry_reward",
    "control_residual_reward", "control_clawback", "control_entry_count",
    "control_residual_count", "route_phi_before", "route_phi_after",
    "route_potential_delta", "route_shaped_reward", "route_positive",
    "route_negative", "route_topology_gain", "route_phase_transition",
    "route_invalidated", "route_terminal_correction", "route_distance_before",
    "route_distance_after", "route_inf_to_finite", "route_safety_rejected",
    "route_phase", "route_target_identity", "route_target_provenance",
)
ITEM_DIAGNOSTICS = (
    "banana_completed", "banana_benefit", "banana_pay", "glue_completed",
    "glue_benefit", "glue_pay", "item_extra",
)


def native_component_schema(length: int) -> dict[str, Any]:
    """Return an explicit index map, including the ABI limitation."""
    if length == 90:
        names = list(NATIVE_COMPONENTS_82) + list(ITEM_DIAGNOSTICS) + ["combined_total"]
        status = "known_90_v1"
    elif length == 83:
        names = list(NATIVE_COMPONENTS_82) + ["combined_total"]
        status = "known_83_v1"
    else:
        names = [f"native_part_{i}" for i in range(length)]
        status = "unknown_length"
    return {"length": length, "status": status, "indices": {n: i for i, n in enumerate(names)}}


def _num(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return sum(_num(r.get("exposure", {}).get(field)) for r in rows)


def read_updates(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON lines and ignore a concurrently written tail."""
    latest: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
            update = int(row["new_update"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        latest[update] = row
    return [latest[k] for k in sorted(latest)]


def _component_sums(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    arrays = [r.get("exposure", {}).get("native_parts") for r in rows]
    arrays = [a for a in arrays if isinstance(a, list)]
    if not arrays:
        return {}, {"length": 0, "status": "未记录", "indices": {}}
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        return {}, {"lengths": sorted(lengths), "status": "inconsistent", "indices": {}}
    n = len(arrays[0])
    schema = native_component_schema(n)
    sums = {name: 0.0 for name in schema["indices"]}
    for array in arrays:
        for name, index in schema["indices"].items():
            sums[name] += _num(array[index])
    return sums, schema


def _reward_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parts, schema = _component_sums(rows)
    abs_total = _sum_field(rows, "absolute_total_reward")
    transport = sum(parts.get(n, 0.0) for n in (
        "transport_shape", "transport_capture", "drop_bonus", "drop_approach_net", "carry_home_net"))
    kill_control = sum(parts.get(n, 0.0) for n in (
        "settled_kill", "settled_shaping", "control_entry_reward",
        "control_residual_reward", "control_clawback"))
    route = parts.get("route_shaped_reward", 0.0)
    penalty = sum(v for k, v in parts.items() if k in {
        "global_safety", "clawback", "blocked_route", "transport_clawback",
        "drop_clawback", "route_negative", "item_clawback",
    } and v < 0)
    shaping = _sum_field(rows, "chain_reward")
    denom = abs_total if abs_total > 0 else None
    fractions = {
        "transport_chain": transport / denom if denom else None,
        "kill_control": kill_control / denom if denom else None,
        "threat_topology_chain": shaping / denom if denom else None,
        "route_pbrs": route / denom if denom else None,
        "penalty": penalty / denom if denom else None,
    }
    return {
        "transport_chain": transport,
        "kill_control": kill_control,
        "threat_topology_chain": shaping,
        "route_pbrs": route,
        "penalty": penalty,
        "absolute_total_reward": abs_total,
        "fractions_of_absolute_total": fractions,
        "native_component_schema": schema,
    }


def aggregate_window(run_name: str, rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    """Aggregate one fixed 40-update window."""
    if not rows:
        raise ValueError("cannot aggregate an empty window")
    first, last = rows[0], rows[-1]
    updates = [int(r["new_update"]) for r in rows]
    exp = {k: _sum_field(rows, k) for k in (
        "learner_bombs_placed", "learner_bomb_requests_rejected", "threat_events",
        "topology_events", "captures", "learner_deaths", "resets",
        "clear_scene_ticks", "clear_scene_resets", "bank_resets", "total_ticks",
    )}
    total_ticks = exp["total_ticks"]
    resets = exp["resets"]
    pool_ticks = [sum(_num(r.get("exposure", {}).get("pool_ticks", [])[i])
                      for r in rows if i < len(r.get("exposure", {}).get("pool_ticks", [])))
                  for i in range(len(OPPONENTS))]
    pool_episodes = [sum(_num(r.get("exposure", {}).get("pool_new_episodes", [])[i])
                         for r in rows if i < len(r.get("exposure", {}).get("pool_new_episodes", [])))
                     for i in range(len(OPPONENTS))]
    components, schema = _component_sums(rows)
    # settled_kill is a reward channel, not an event count.
    kills = _sum_field(rows, 'learner_kills') if all('learner_kills' in r.get('exposure', {}) for r in rows) else None
    finite = all(bool(r.get("finite")) for r in rows)
    loss_rows = [r.get("loss") for r in rows if isinstance(r.get("loss"), list)]
    loss_vector = "未记录"
    loss_schema = "未记录"
    if loss_rows and len({len(x) for x in loss_rows}) == 1:
        loss_vector = [sum(_num(x[i]) for x in loss_rows) / len(loss_rows) for i in range(len(loss_rows[0]))]
        loss_schema = ['loss','ppo','value','entropy','approx_kl','clip_fraction','gradient_norm'] if len(loss_vector)==7 else (
            ['loss','ppo','value','entropy','approx_kl','clip_fraction','gradient_norm','ability_head_gradient_norm'] if len(loss_vector)==8 else ['未记录']*len(loss_vector))
    cycles = sum(_num(r.get("cycle_seconds")) for r in rows)
    actor_l2 = [_num(r.get("update_l2", {}).get("actor")) for r in rows if "actor" in r.get("update_l2", {})]
    critic_l2 = [_num(r.get("update_l2", {}).get("critic")) for r in rows if "critic" in r.get("update_l2", {})]
    total_env = sum(_num(r.get("actual_environment_steps_this_update")) for r in rows)
    opponent = {
        name: {
            "pool_ticks": pool_ticks[i], "new_episodes": pool_episodes[i],
            "wins": "未记录", "kills": "未记录", "deaths": "未记录", "delivery": "未记录",
        } for i, name in enumerate(OPPONENTS)
    }
    diagnostic_ticks = _sum_field(rows, 'policy_diagnostic_ticks')
    entropy = _sum_field(rows, 'policy_entropy_sum') / diagnostic_ticks if diagnostic_ticks else (
        loss_vector[3] if isinstance(loss_vector,list) and len(loss_vector) in (7,8) else '未记录')
    probability = _sum_field(rows, 'bomb_probability_sum') / diagnostic_ticks if diagnostic_ticks else '未记录'
    argmax = _sum_field(rows, 'bomb_argmax_count') / diagnostic_ticks if diagnostic_ticks else '未记录'
    missing = ['opponent_terminal_outcomes']
    if not diagnostic_ticks:missing += ['collector_policy_entropy', 'bomb_action_probability', 'bomb_argmax_fraction']
    if entropy=='未记录':missing.append('policy_entropy')
    if kills is None:missing.append('learner_kill_events')
    reward = _reward_summary(rows)
    return {
        "run": run_name, "window": window, "first_update": min(updates),
        "last_update": max(updates), "update_count": len(rows),
        "exposure": exp,
        "per_100_tick": {"bombs_placed": exp["learner_bombs_placed"] * 100 / total_ticks if total_ticks else None,
                          "requests_rejected": exp["learner_bomb_requests_rejected"] * 100 / total_ticks if total_ticks else None,
                          "threat_events": exp["threat_events"] * 100 / total_ticks if total_ticks else None,
                          "topology_events": exp["topology_events"] * 100 / total_ticks if total_ticks else None,
                          "captures": exp["captures"] * 100 / total_ticks if total_ticks else None,
                          "deaths": exp["learner_deaths"] * 100 / total_ticks if total_ticks else None,
                          "kills": kills * 100 / total_ticks if kills is not None and total_ticks else None},
        "avg_episode_ticks": total_ticks / resets if resets else None,
        "clear_scene_ratio": exp["clear_scene_resets"] / resets if resets else None,
        "clear_scene_tick_ratio": exp["clear_scene_ticks"] / total_ticks if total_ticks else None,
        "bank_ratio": exp["bank_resets"] / resets if resets else None,
        "reward": reward,
        "policy_entropy": entropy, "bomb_action_probability": probability,
        "policy_entropy_source": 'masked_live_collector' if diagnostic_ticks else 'optimizer_minibatch' if entropy!='未记录' else '未记录',
        "bomb_argmax_fraction": argmax, "policy_diagnostic_ticks": diagnostic_ticks,
        "optimizer_minibatch_entropy": loss_vector[3] if isinstance(loss_vector,list) and len(loss_vector) in (7,8) else '未记录',
        "loss_vector": loss_vector, "loss_vector_schema": loss_schema,
        "loss_schema_status": "partial/未完整记录",
        "opponents": opponent,
        "env_per_second": total_env / cycles if cycles else None,
        "update_l2": {"actor": sum(actor_l2) / len(actor_l2) if actor_l2 else None,
                       "critic": sum(critic_l2) / len(critic_l2) if critic_l2 else None},
        "finite": finite, "missing_data": missing,
        "unix_last": _num(last.get("unix")),
    }


def _alert(run: str, window: int, code: str, detail: str) -> dict[str, Any]:
    return {"run": run, "window": window, "code": code, "detail": detail,
            "severity": 'DATA_GAP' if code.endswith('UNAVAILABLE') else 'ANOMALY',
            "action": "记录/展示；不自动停训"}


def detect_alerts(windows: list[dict[str, Any]], active: bool = False, now: float | None = None) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if not windows:
        return alerts
    baseline = windows[0]
    for i, row in enumerate(windows):
        if not row["finite"]:
            alerts.append(_alert(row["run"], row["window"], "NONFINITE", "finite=false"))
        if row["env_per_second"] is not None and row["env_per_second"] < 15000:
            alerts.append(_alert(row["run"], row["window"], "LOW_ENV_PER_SECOND", f"env/s={row['env_per_second']:.2f}"))
        if row["reward"]["fractions_of_absolute_total"].get("threat_topology_chain") is not None and abs(row["reward"]["fractions_of_absolute_total"]["threat_topology_chain"]) > .30:
            alerts.append(_alert(row["run"], row["window"], "SHAPING_OVER_30_PERCENT", "threat/topology/chain exceeds 30% absolute reward"))
        if len(windows) >= 3 and i >= 2:
            tail = windows[i-2:i+1]
            base_bombs = baseline["per_100_tick"].get("bombs_placed")
            vals = [x["per_100_tick"].get("bombs_placed") for x in tail]
            if base_bombs and all(v is not None and v < .8 * base_bombs for v in vals):
                alerts.append(_alert(row["run"], row["window"], "LOW_BOMBS_THREE_WINDOWS", "落泡数低于起点窗口20%连续3窗"))
            base_deaths = baseline["per_100_tick"].get("deaths")
            death_vals = [x["per_100_tick"].get("deaths") for x in tail]
            if base_deaths and all(v is not None and v > 1.5 * base_deaths for v in death_vals):
                alerts.append(_alert(row["run"], row["window"], "DEATH_RATE_UP_50_PERCENT", "死亡率高于起点窗口50%连续3窗"))
        if i > 0:
            previous = windows[i - 1]
            captures_now = row["per_100_tick"].get("captures")
            captures_before = previous["per_100_tick"].get("captures")
            kills_now = row["per_100_tick"].get("kills")
            kills_before = previous["per_100_tick"].get("kills")
            if all(isinstance(v, (int, float)) for v in (captures_now, captures_before, kills_now, kills_before)):
                if captures_now > captures_before and kills_now < kills_before:
                    alerts.append(_alert(row["run"], row["window"], "DELIVERY_UP_KILLS_DOWN", "交付/100tick上升且击杀/100tick下降"))
        numeric_entropy = [x.get("policy_entropy") for x in windows[max(0, i-2):i+1]]
        if len(numeric_entropy) == 3 and all(isinstance(v, (int, float)) for v in numeric_entropy):
            if numeric_entropy[-1] <= .5 * numeric_entropy[0]:
                alerts.append(_alert(row["run"], row["window"], "ENTROPY_DOWN_50_PERCENT", "策略熵三窗下降50%"))
        numeric_prob = row.get("bomb_action_probability")
        if isinstance(numeric_prob, (int, float)) and numeric_prob < 1e-3:
            alerts.append(_alert(row["run"], row["window"], "BOMB_PROBABILITY_NEAR_ZERO", "放泡动作概率接近0"))
        if i == len(windows)-1 and row["unix_last"] and active and now is not None and now - row["unix_last"] > 900:
            alerts.append(_alert(row["run"], row["window"], "NO_UPDATE_15_MINUTES", "active run last update超过15分钟"))
    # These checks are intentionally explicit: absent collector fields are not zero.
    for code, field in (("ENTROPY_UNAVAILABLE", "policy_entropy"), ("BOMB_PROBABILITY_UNAVAILABLE", "bomb_action_probability")):
        if all(row.get(field) == "未记录" for row in windows):
            alerts.append(_alert(windows[-1]["run"], windows[-1]["window"], code, "collector未记录，无法判定趋势"))
    return alerts


def active_run_names() -> set[str]:
    found=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if not any(a.startswith(b'scripts.bomb_threat_') and a.endswith(b'train') for a in args):continue
            if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            for arg in args:
                if arg.startswith(b'runs/bomb_threat_v1/protocols/'):
                    found.add(Path(arg.decode()).stem)
        except OSError:continue
    return found


def collect(root: Path = BASE) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_windows: list[dict[str, Any]] = []
    all_alerts: list[dict[str, Any]] = []
    active = active_run_names()
    for path in sorted((root / "formal").glob("*/updates.jsonl")):
        rows = read_updates(path)
        if not rows:
            continue
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(int(row["new_update"]) - 1) // 40 + 1].append(row)
        run_windows = [aggregate_window(path.parent.name, groups[k], k) for k in sorted(groups)]
        all_windows.extend(run_windows)
        all_alerts.extend(detect_alerts(run_windows, path.parent.name in active, time.time()))
    return all_windows, all_alerts


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def refresh(root: Path = BASE):
    from scripts.population_run_state import atomic_json, lock
    guard=lock(root/'TRAIN_MONITOR.lock')
    windows, alerts = collect(root)
    active=active_run_names()
    latest={w['run']:w['window'] for w in windows}
    for alert in alerts:alert['scope']='CURRENT' if alert['run'] in active and alert['window']==latest.get(alert['run']) else 'HISTORICAL'
    for window in windows:window['scope']='CURRENT' if window['run'] in active and window['window']==latest.get(window['run']) else 'HISTORICAL'
    _write_jsonl(root / "TRAIN_MONITOR.jsonl", windows)
    # Deduplicate on reruns while preserving the latest detail for each event.
    unique = {(a["run"], a["window"], a["code"]): a for a in alerts}
    _write_jsonl(root / "TRAIN_ALERTS.jsonl", unique.values())
    receipt = {"status": "COMPLETE", "windows": len(windows), "alerts": len(unique),
               "runs": sorted({w["run"] for w in windows}), "generated_unix": time.time(),
               "alert_counts_by_severity": dict(Counter(a['severity'] for a in unique.values())),
               "alert_counts_by_scope": dict(Counter(a['scope'] for a in unique.values())),
               "alert_counts_by_scope_and_severity": dict(Counter(a['scope']+'/'+a['severity'] for a in unique.values())),
               "current_latest": [dict(run=w['run'],update=w['last_update'],env_per_second=w['env_per_second']) for w in windows if w['scope']=='CURRENT' and not any(v['run']==w['run'] and v['last_update']>w['last_update'] for v in windows)],
               "outputs": ["TRAIN_MONITOR.jsonl", "TRAIN_ALERTS.jsonl"]}
    atomic_json(root / 'TRAIN_MONITOR_STATUS.json', receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BASE)
    args = parser.parse_args(argv)
    receipt=refresh(args.root)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
