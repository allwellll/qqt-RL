"""CPU-only A reward-alignment and same-RNG native-JAX diagnostics.

The rollout uses the sealed native training transition through
``combat_item_engine`` and records the real component vector.  No replay from
this diagnostic is fed to training.  The run is deliberately small and is a
measurement artifact, not a capability gate.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("JAXBOMB_RULE", "bun")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

from scripts import carrier_mechanism_engine as E
from scripts import combat_item_engine as T
from scripts import combat_item_events as item_events
from scripts import je10_ops as ops
from scripts import je09_bun_scenes as scenes
from scripts.population_run_state import atomic_json, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_validation_20261025"
STEPS = 64
EPISODES = 8
SEED = 2026102860

CHECKPOINTS = {
    "parent": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102751/u064/state.pkl",
    "r2_u256": "runs/ppo_scale_20261025/r2_longrun/formal/ppo_scaled_batch_continuation_s2026102751/u000256/state.pkl",
}


def component_schema(base_names, item_names):
    """Match the collector's metric append order and preserve one total column."""
    base_names = tuple(base_names)
    if not base_names or base_names[-1] != "combined_total":
        raise ValueError("base reward schema must end in combined_total")
    names = (*base_names[:-1], *tuple(item_names), "combined_total")
    if len(set(names)) != len(names):
        raise ValueError("reward component schema contains duplicate labels")
    return names


def load_params(path: str):
    with (R / path).open("rb") as stream:
        state = pickle.load(stream)
    return state["params"], {
        "state_sha256": sha(R / path),
        "update": state.get("update"),
        "seed": state.get("seed"),
    }


def select_tree(mask, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b),
        new, old,
    )


def rollout(params, steps=STEPS):
    keys = jax.random.split(jax.random.PRNGKey(SEED), EPISODES)
    states, aux = E.initial(keys, jnp.array(0, jnp.int32))
    aux = dict(aux, tactics=item_events.state(EPISODES))
    names = component_schema(ops.COMPONENTS, item_events.METRICS)

    def body(carry, _):
        st, ax, key, ended = carry
        next_key, env_key, actions, model_data = E.choose(params, st, ax, key)
        after, next_aux, reward, done, info, parts = T.transition(st, ax, actions, env_key)
        active = ~ended
        # Hold terminal rows at their last native state, exactly as the frozen
        # collector does; no auto-reset or synthetic carry is introduced.
        after = select_tree(ended, st, after)
        next_aux = select_tree(ended, ax, next_aux)
        next_key = select_tree(ended, key, next_key)
        ended_next = ended | done
        cc = next_aux["control_credit"]
        route = next_aux["route_pbrs"]
        transport = next_aux["transport"]
        drop = next_aux["drop"]
        tactics = next_aux["tactics"]
        event_names = (
            "steal", "pickup_loose", "capture", "death", "drop",
            "surviving_kill", "native_transfer", "return_bun",
        )
        event_columns = []
        for name in event_names:
            value = jnp.asarray(info.get(name, jnp.zeros_like(done)))
            if value.ndim == 1:
                p0 = value
            elif value.ndim == 2:
                p0 = value[:, 0]
            else:
                p0 = value[:, 0].any(axis=-1)
            event_columns.append(p0)
        event = jnp.stack(event_columns, axis=-1)
        phase = route.get("phase", jnp.zeros((EPISODES, 2), jnp.int32))[:, 0]
        identity = route.get("identity", jnp.zeros((EPISODES, 2), jnp.int32))[:, 0]
        carry = after.bun_carried[:, 0]
        window = (~st.alive[:, 1] | (st.trapped[:, 1] > 0)).astype(jnp.int32)
        ledger_names = (
            "control_gross_p0", "control_balance_p0", "route_phi_p0", "route_offset_p0",
            "transport_balance_p0", "transport_pending_capture_p0", "transport_episode_gross_p0",
            "drop_balance_origin0", "drop_balance_origin1",
            "drop_pending_capture_origin0", "drop_pending_capture_origin1",
            "banana_balance_p0", "banana_legacy_balance_p0", "banana_gross_p0",
            "glue_balance_p0", "glue_legacy_balance_p0", "glue_gross_p0",
        )
        ledger_values = (
            cc["gross"][:, 0], cc["balance"][:, 0], route["phi"][:, 0], route["offset"][:, 0],
            transport["balance"][:, 0], transport["pending_capture"][:, 0], transport["episode_gross"][:, 0],
            drop["balance"][:, 0], drop["balance"][:, 1],
            drop["pending_capture"][:, 0], drop["pending_capture"][:, 1],
            tactics["banana_balance"][:, 0], tactics["banana_legacy_balance"][:, 0], tactics["banana_gross"][:, 0],
            tactics["glue_balance"][:, 0], tactics["glue_legacy_balance"][:, 0], tactics["glue_gross"][:, 0],
        )
        ledger = jnp.stack(ledger_values, axis=-1)
        # p0 only is the measured learner seat; the full component vector is
        # retained so control/route/item/pending balances remain separable.
        record = (
            jnp.where(active, reward[:, 0], 0.),
            jnp.where(active[:, None], parts[:, 0], 0.), actions[:, 0], active,
            event, phase, identity, carry, window,
            cc["gross"][:, 0], cc["balance"][:, 0], cc["event_id"][:, 0],
            cc["owner"][:, 0], route["phi"][:, 0], ledger,
            jnp.where(active, done, False),
        )
        return (after, next_aux, next_key, ended_next), record

    # JIT/scan keeps A deterministic while leaving the checkpoint params
    # untouched.  The first call includes compilation; its time is recorded.
    scan = jax.jit(lambda s, a, k: jax.lax.scan(body, (s, a, k, jnp.zeros((EPISODES,), bool)), None, length=steps))
    (end_state, end_aux, end_key, ended), records = scan(states, aux, keys)
    jax.block_until_ready(end_state.t)
    arrays = [np.asarray(x) for x in records]
    return names, arrays


def summarize(name: str, meta: dict, names: list[str], arrays: list[np.ndarray]) -> dict:
    reward, parts, actions, active, event, phase, identity, carry, window, gross, balance, event_id, owner, distance, ledger, terminal = arrays
    num_steps = int(reward.shape[0])
    num_episodes = int(reward.shape[1])
    valid = active.astype(bool)
    valid_count = int(valid.sum())
    flat_parts = parts[valid]
    if flat_parts.ndim != 2 or flat_parts.shape[-1] != len(names):
        raise ValueError("collector reward vector width differs from component schema")
    flat_reward = reward[valid]
    # component rows are genuine training-time outputs, unlike the JS
    # reconstruction used by the separate B audit.
    component_sum = flat_parts.sum(axis=0).tolist() if valid_count else []
    events = event[valid]
    event_sum = events.sum(axis=0).tolist() if valid_count else []
    # discounted and undiscounted returns, with no bootstrap term in this
    # finite diagnostic horizon.  Terminal rows are separated by ``active``.
    undiscounted = reward.sum(axis=0).tolist()
    discounted = np.zeros(num_episodes, np.float64)
    for t in range(num_steps - 1, -1, -1):
        discounted = reward[t] + 0.995 * discounted
    per_episode = []
    for episode in range(num_episodes):
        episode_valid = valid[:, episode]
        episode_terminal = terminal[:, episode] & episode_valid
        phase_actions = {}
        for phase_code in np.unique(phase[:, episode][episode_valid]):
            selected = episode_valid & (phase[:, episode] == phase_code)
            phase_actions[str(int(phase_code))] = {
                "ticks": int(selected.sum()),
                "move": np.bincount(actions[selected, episode, 0], minlength=5).tolist(),
                "ability": np.bincount(actions[selected, episode, 1], minlength=3).tolist(),
            }
        per_episode.append({
            "episode_index": episode,
            "valid_ticks": int(episode_valid.sum()),
            "terminal_transition_count": int(episode_terminal.sum()),
            "undiscounted_return": float(reward[:, episode].sum()),
            "discounted_return_gamma_0.995": float(discounted[episode]),
            "component_sum": parts[:, episode][episode_valid].sum(axis=0).tolist(),
            "events": event[:, episode][episode_valid].sum(axis=0).tolist(),
            "phase_action_counts": phase_actions,
            "final_ledger_balances": dict(zip(
                ("control_gross_p0", "control_balance_p0", "route_phi_p0", "route_offset_p0",
                 "transport_balance_p0", "transport_pending_capture_p0", "transport_episode_gross_p0",
                 "drop_balance_origin0", "drop_balance_origin1", "drop_pending_capture_origin0",
                 "drop_pending_capture_origin1", "banana_balance_p0", "banana_legacy_balance_p0",
                 "banana_gross_p0", "glue_balance_p0", "glue_legacy_balance_p0", "glue_gross_p0"),
                ledger[-1, episode].tolist())),
            "artificial_horizon_truncated": bool(episode_valid.sum() == valid.shape[0] and episode_terminal.sum() == 0),
        })
    return {
        "checkpoint": name,
        **meta,
        "engine": "native JAX training transition; fixed rule bot; teacher off",
        "episodes": EPISODES,
        "steps": num_steps,
        "valid_transition_count": valid_count,
        "component_names": names,
        "component_sum_p0": component_sum,
        "event_names": ["steal", "pickup_loose", "capture", "death", "drop", "surviving_kill", "native_transfer", "return_bun"],
        "event_sum_p0": event_sum,
        "per_episode": per_episode,
        "undiscounted_return_p0": undiscounted,
        "discounted_return_p0_gamma_0.995": discounted.tolist(),
        "action_counts_move": np.bincount(actions[..., 0].reshape(-1), minlength=5).tolist(),
        "action_counts_ability": np.bincount(actions[..., 1].reshape(-1), minlength=3).tolist(),
        "window_ticks": int(window[valid].sum()),
        "carry_ticks": int((carry[valid] >= 0).sum()),
        "mean_control_gross": float(gross[valid].mean()),
        "mean_control_balance": float(balance[valid].mean()),
        "max_control_event_id": int(event_id[valid].max()) if valid_count else 0,
        "owner_nonnegative_ticks": int((owner[valid] >= 0).sum()),
        "route_potential_mean": float(distance[valid].mean()),
        "phase_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(phase[valid], return_counts=True))},
        "target_identity_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(identity[valid], return_counts=True))},
        "bootstrap_status": "no V bootstrap used; terminal transitions are counted separately; artificial truncations are labeled per episode; finite-window MC is not a full-return or V-accuracy estimate",
        "per_episode_ledger_component_names": [
            "control_gross_p0", "control_balance_p0", "route_phi_p0", "route_offset_p0",
            "transport_balance_p0", "transport_pending_capture_p0", "transport_episode_gross_p0",
            "drop_balance_origin0", "drop_balance_origin1", "drop_pending_capture_origin0",
            "drop_pending_capture_origin1", "banana_balance_p0", "banana_legacy_balance_p0",
            "banana_gross_p0", "glue_balance_p0", "glue_legacy_balance_p0", "glue_gross_p0",
        ],
        "teacher_applied": False,
        "training_eligible": False,
    }


def horizon_summary_rows(rows: list[dict], extended_paths: dict[str, Path]) -> list[dict]:
    """Read completed, separately executed extended-horizon CPU artifacts."""
    extended = []
    for checkpoint, path in extended_paths.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if payload.get("schema") != "qqt.expert-validation-reward-alignment-horizon/v1":
            raise ValueError(f"unexpected horizon artifact schema: {path.name}")
        if payload.get("checkpoint") != checkpoint or not isinstance(payload.get("horizon"), int):
            raise ValueError(f"horizon artifact identity mismatch: {path.name}")
        result = payload.get("result", {})
        if result.get("checkpoint") != checkpoint or result.get("steps") != payload["horizon"]:
            raise ValueError(f"horizon result does not match its declared identity: {path.name}")
        resolved_path = path.resolve()
        try:
            artifact_path = resolved_path.relative_to(R).as_posix()
        except ValueError:
            artifact_path = path.name
        extended.append({
            "checkpoint": checkpoint,
            "state_sha256": result.get("state_sha256"),
            "horizon": payload["horizon"],
            "valid_transition_count": result.get("valid_transition_count"),
            "terminal_transition_count": [ep["terminal_transition_count"] for ep in result["per_episode"]],
            "artificial_horizon_truncated": [ep["artificial_horizon_truncated"] for ep in result["per_episode"]],
            "event_sum_p0": result.get("event_sum_p0"),
            "undiscounted_return_p0": result.get("undiscounted_return_p0"),
            "discounted_return_p0_gamma_0.995": result.get("discounted_return_p0_gamma_0.995"),
            "no_v_bootstrap": "no V bootstrap used" in result.get("bootstrap_status", ""),
            "artifact": artifact_path,
            "artifact_sha256": sha(path),
        })
    return extended


def build_horizon_sensitivity(rows: list[dict], extended_paths: dict[str, Path]) -> dict:
    base_rows = [
        {"checkpoint": row["checkpoint"], "horizon": row["steps"],
         "undiscounted_return_p0": row["undiscounted_return_p0"],
         "discounted_return_p0_gamma_0.995": row["discounted_return_p0_gamma_0.995"],
         "terminal_transition_count": [ep["terminal_transition_count"] for ep in row["per_episode"]],
         "artificial_horizon_truncated": [ep["artificial_horizon_truncated"] for ep in row["per_episode"]]}
        for row in rows
    ]
    return {
        "schema": "qqt.expert-validation-reward-alignment-horizon/v1",
        "protocol": {
            "registered_base_horizon": STEPS,
            "same_initial_keys_and_env_rng": True,
            "teacher_off": True,
            "no_replay_to_training": True,
            "purpose": "separate terminal and artificial truncation accounting; no V bootstrap is used",
        },
        "base_rows": base_rows,
        "extended_rows": horizon_summary_rows(rows, extended_paths),
        "source_sha256": {
            "scripts/expert_validation_diagnostics.py": sha(R / "scripts/expert_validation_diagnostics.py"),
            "A_reward_alignment.json": sha(B / "A_reward_alignment.json"),
        },
        "does_not_control_gpu": True,
    }


def main() -> None:
    T.configure()
    rows = []
    for name, path in CHECKPOINTS.items():
        params, meta = load_params(path)
        names, arrays = rollout(params)
        rows.append(summarize(name, meta, names, arrays))
    alignment = {
        "schema": "qqt.expert-validation-reward-alignment/v2",
        "protocol": {
            "seed": SEED,
            "episodes": EPISODES,
            "steps": STEPS,
            "same_initial_keys_and_env_rng": True,
            "same_native_rule_engine": True,
            "teacher_off": True,
            "no_replay_to_training": True,
        },
        "rows": rows,
        "schema_correction": {
            "previous_artifact": "runs/expert_validation_20261025/A_reward_alignment_COMPONENT_SCHEMA_MISMATCH.json",
            "previous_artifact_sha256": sha(B / "A_reward_alignment_COMPONENT_SCHEMA_MISMATCH.json"),
            "corrected_component_schema": "je10_ops.COMPONENTS[:-1] + combat_item_events.METRICS + combined_total",
            "corrected_component_width": 90,
            "same_registered_seed_initials_and_rng": True,
            "teacher_off": True,
            "training_eligible": False,
        },
        "source_sha256": {
            "scripts/carrier_mechanism_engine.py": sha(R / "scripts/carrier_mechanism_engine.py"),
            "scripts/combat_item_engine.py": sha(R / "scripts/combat_item_engine.py"),
            "scripts/je10_ops.py": sha(R / "scripts/je10_ops.py"),
            "scripts/expert_validation_diagnostics.py": sha(R / "scripts/expert_validation_diagnostics.py"),
        },
        "does_not_control_gpu": True,
    }
    atomic_json(B / "A_reward_alignment.json", alignment)

    atomic_json(B / "A_horizon_sensitivity.json", build_horizon_sensitivity(rows, {
        "parent": B / "A_horizon_parent_128.json",
        "r2_u256": B / "A_horizon_r2_u256_128.json",
    }))


if __name__ == "__main__":
    main()
