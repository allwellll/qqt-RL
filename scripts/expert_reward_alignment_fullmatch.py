"""Fixed natural full-match reward alignment on the native JAX training engine.

This is CPU-only diagnostics.  Each real trajectory is scored both by the
legacy reward emitted by the native engine and by the frozen D minimal adapter;
no diagnostic trajectory is fed to training.
"""
from __future__ import annotations

import json
import hashlib
import os
import pickle
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                  JAX_PLATFORM_NAME="cpu", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", JAXBOMB_RULE="bun")

import jax
import jax.numpy as jnp
import numpy as np

from scripts import carrier_mechanism_engine as E
from scripts import combat_item_engine as T
from scripts import combat_item_events as item_events
from scripts import expert_reward_ablation as reward_adapter
from scripts import expert_reward_ablation_prepare as prep
from scripts import je10_control_credit as control_credit
from scripts import je09_bun_scenes as scenes
from scripts import dagger_rl_r4_train as provenance
from scripts import passive_evaluation_metrics as passive_metrics
from scripts.population_run_state import atomic_json, lock, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/expert_reward_20261025/D_ablation_v1"
OUT = Path(os.environ.get(
    "FULLMATCH_OUTPUT",
    str(B / "fullmatch_reward_alignment_v2")))
MANIFEST = OUT / "MANIFEST.json"
SEED_START = int(os.environ.get("FULLMATCH_SEED_START", "2026102680"))
GAMES_PER_MODE = int(os.environ.get("FULLMATCH_GAMES_PER_MODE", "32"))
BATCH_SIZE = 8
MAX_TICKS = 2560
CHUNK = 128
GAMMA = 0.995
ALL_MODELS = {
    "parent_r1_s2751_u064": {
        "state": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102751/u064/state.pkl",
        "checkpoint": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102751/u064",
        "run_seed": 2026102751,
    },
    "parent_r1_s2752_u064": {
        "state": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102752/u064/state.pkl",
        "checkpoint": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102752/u064",
        "run_seed": 2026102752,
    },
    "parent_r1_s2753_u064": {
        "state": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102753/u064/state.pkl",
        "checkpoint": "runs/ppo_scale_20261025/r1_longrun/formal/ppo_scaled_batch_s2026102753/u064",
        "run_seed": 2026102753,
    },
    "legacy_s2851_u320": {
        "state": "runs/expert_reward_20261025/D_ablation_v1/formal/legacy_s2026102851/u000320/state.pkl",
        "checkpoint": "runs/expert_reward_20261025/D_ablation_v1/formal/legacy_s2026102851/u000320",
        "run_seed": 2026102851,
    },
    "minimal_s2851_u064": {
        "state": "runs/expert_reward_20261025/D_ablation_v1/formal/minimal_s2026102851/u000064/state.pkl",
        "checkpoint": "runs/expert_reward_20261025/D_ablation_v1/formal/minimal_s2026102851/u000064",
        "run_seed": 2026102851,
    },
    "r2_early_stopped_s2751_u1664": {
        "state": "runs/ppo_scale_20261025/r2_longrun/formal/ppo_scaled_batch_continuation_s2026102751/u001664/state.pkl",
        "checkpoint": "runs/ppo_scale_20261025/r2_longrun/formal/ppo_scaled_batch_continuation_s2026102751/u001664",
        "run_seed": 2026102751,
    },
}
_requested_models = os.environ.get("FULLMATCH_MODELS", "").strip()
MODELS = ({name: ALL_MODELS[name] for name in _requested_models.split(",") if name}
          if _requested_models else ALL_MODELS)
unknown_models = set(_requested_models.split(",")) - set(ALL_MODELS) if _requested_models else set()
if not MODELS or unknown_models:
    raise ValueError(f"invalid FULLMATCH_MODELS selection: {sorted(unknown_models)}")
EVENT_NAMES = ("capture", "death", "steal", "pickup_loose", "drop",
               "surviving_kill", "native_transfer", "return_bun")
LEDGER_NAMES = (
    "transport_balance_p0", "transport_pending_capture_p0",
    "transport_episode_gross_p0", "control_balance_p0", "control_gross_p0",
    "route_phi_p0", "route_offset_p0",
)
EXCLUSIVE_LEGACY_TERMS = (
    "lesson_task", "global_safety", "settled_kill", "settled_shaping",
    "clawback", "blocked_route", "transport_shape", "transport_clawback", "transport_capture",
    "drop_bonus", "drop_clawback", "drop_approach_net", "carry_home_net",
    # phase_reward returns delta separately from clawback.  The aggregate
    # potential_clawback metric is an additive reward channel; the per-phase
    # *_clawback fields are diagnostic mirrors and must not be counted again.
    "potential_clawback", "bridge_reward", "bridge_clawback", "control_entry_reward",
    "control_residual_reward", "control_clawback", "item_positive",
    "item_clawback", "route_shaped_reward",
)
EXCLUSIVE_MINIMAL_TERMS = (
    "delivery", "kill", "death", "control_entry", "control_residual", "route_pbrs",
)
REPORTED_TOTAL_COMPONENTS = ("combined_total",)
DECOMPOSITION_ATOL = 1e-4

# Fixed passive trace ABI. Values are numeric so the JAX scan stays a pure
# observation pass; conversion to named rows happens after the scan.
TELEMETRY_NAMES = (
    "tick", "phase_code", "enemy_alive_before", "enemy_alive_after",
    "enemy_trapped_before", "enemy_respawn_before", "enemy_death",
    "enemy_native_trap_enter", "window_cause_owner", "enemy_distance_manhattan",
    "own_carried_before", "own_carried_after", "own_capture", "own_return_bun",
    "own_death", "own_drop", "own_trapped_after", "stock_pickup",
    "loose_enemy_pickup", "loose_own_pickup", "native_transfer", "own_move",
    "own_ability", "move_entropy", "ability_entropy", "move_prob_0",
    "move_prob_1", "move_prob_2", "move_prob_3", "move_prob_4",
    "ability_prob_0", "ability_prob_1", "ability_prob_2", "bomb_request",
    "bomb_legal", "bomb_placed", "item_inventory", "item_release_opportunity",
    "item_release_request", "item_placed", "item_hit_self", "item_hit_enemy",
    "banana_hit_proxy", "glue_hit_proxy", "own_position_delta",
    "home_distance_manhattan_before", "home_distance_manhattan_after",
    "enemy_stock_distance_before",
    "legacy_reward", "minimal_reward", "winner", "active", "done",
    "death_cause_owner", "native_syrup_cause_owner",
)


def _masked_entropy(logits, mask):
    probs = jax.nn.softmax(jnp.where(mask, logits, -1e9), axis=-1)
    return probs, -jnp.sum(jnp.where(probs > 0, probs * jnp.log(jnp.maximum(probs, 1e-20)), 0.), axis=-1)


def _passive_snapshot(st, after, info, actions, policy, mm, bm,
                      legacy_reward, minimal_reward, active, done):
    """Build a passive numeric trace row; no state or action is modified."""
    move_probs, ability_probs = policy
    n = st.t.shape[0]
    p0 = st.pos[:, 0]
    p1 = st.pos[:, 1]
    p0_after = after.pos[:, 0]
    p0_cell = jnp.clip(p0.astype(jnp.int32), jnp.array([0, 0]), jnp.array([12, 14]))
    p0_after_cell = jnp.clip(p0_after.astype(jnp.int32), jnp.array([0, 0]), jnp.array([12, 14]))
    field_item_after_blast = jnp.where(
        after.core.blast_linger > 0, 0, st.field_item)
    field_item = field_item_after_blast[jnp.arange(n), p0_cell[:, 0], p0_cell[:, 1]]
    held = st.held_item[:, 0]
    placement_cell_open = (
        (field_item == 0)
        & ~st.wall[jnp.arange(n), p0_cell[:, 0], p0_cell[:, 1]]
        & ~st.brick[jnp.arange(n), p0_cell[:, 0], p0_cell[:, 1]]
        & (st.fuse[jnp.arange(n), p0_cell[:, 0], p0_cell[:, 1]] == 0))
    item_release_legal = ((held > 0) & st.alive[:, 0] & (st.trapped[:, 0] == 0)
                          & bm[:, 2] & placement_cell_open)
    item_placed = (actions[:, 0, 1] == 2) & item_release_legal
    item_hit_self = info["trap_hit"][:, 0] & (info["trap_owner"][:, 0] == 0)
    item_hit_enemy = info["trap_hit"][:, 1] & (info["trap_owner"][:, 1] == 0)
    # A trap hit plus the resulting native status identifies the effect kind.
    # It is an observed hit proxy, not a complete item-chain outcome or causal
    # net-benefit estimate.
    banana_hit = item_hit_enemy & (after.move_status[:, 1] == E.env.STATUS_SLIDE)
    glue_hit = item_hit_enemy & (after.move_status[:, 1] == E.env.STATUS_SLOW)
    free = st.alive[:, 1] & (st.trapped[:, 1] <= 0)
    carrying = st.bun_carried[:, 0] >= 0
    controlled = (~st.alive[:, 1]) | (st.trapped[:, 1] > 0)
    phase = jnp.where(carrying, 2, jnp.where(controlled, 1,
                                             jnp.where(free, 0, 3)))
    own_cause = info["causal_kill"][:, 0] | (
        info["native_trap_popper"][:, 1] == 0)
    enemy_cause = info["own_bomb_defeat"][:, 1]
    cause_owner = jnp.where(own_cause, 0, jnp.where(enemy_cause, 1, -1))
    trap_sources = jnp.where(after.trapped_causal[:, 1].any(-1, keepdims=True),
                             after.trapped_causal[:, 1], after.trapped_source[:, 1])
    trap_owner = jnp.where(trap_sources.sum(-1) == 1,
                           jnp.argmax(trap_sources, -1), -1)
    death_sources = info['causal_death_source'][:, 1]
    death_owner = jnp.where(death_sources.sum(-1) == 1,
                            jnp.argmax(death_sources, -1), -1)
    home = E.env._BUN_BASES[0].astype(jnp.float32) + 1.0
    enemy_base = E.env._BUN_BASES[1].astype(jnp.float32) + 1.0
    before_home = jnp.abs(p0 - home).sum(-1)
    after_home = jnp.abs(p0_after - home).sum(-1)
    before_stock = jnp.abs(p0 - enemy_base).sum(-1)
    move_entropy = -jnp.sum(jnp.where(move_probs > 0,
                                      move_probs * jnp.log(jnp.maximum(move_probs, 1e-20)), 0.), -1)
    ability_entropy = -jnp.sum(jnp.where(ability_probs > 0,
                                         ability_probs * jnp.log(jnp.maximum(ability_probs, 1e-20)), 0.), -1)
    # Keep all booleans as float32 in the trace for a stable, serializable ABI.
    values = [
        st.t.astype(jnp.float32), phase.astype(jnp.float32), st.alive[:, 1], after.alive[:, 1],
        st.trapped[:, 1].astype(jnp.float32), st.bun_respawn[:, 1].astype(jnp.float32),
        info["death"][:, 1], info["native_trap_enter"][:, 1], cause_owner.astype(jnp.float32),
        jnp.abs(p0 - p1).sum(-1), st.bun_carried[:, 0].astype(jnp.float32),
        after.bun_carried[:, 0].astype(jnp.float32), info["capture"][:, 0],
        info["return_bun"][:, 0], info["death"][:, 0], info["drop"][:, 0],
        after.trapped[:, 0].astype(jnp.float32), info["steal"][:, 0],
        info["pickup_loose"][:, 0] & (info["pickup_origin"][:, 0] == 1),
        info["pickup_loose"][:, 0] & (info["pickup_origin"][:, 0] == 0),
        info["native_transfer"][:, 0], actions[:, 0, 0].astype(jnp.float32),
        actions[:, 0, 1].astype(jnp.float32), move_entropy, ability_entropy,
        *[move_probs[:, i] for i in range(move_probs.shape[-1])],
        *[ability_probs[:, i] for i in range(ability_probs.shape[-1])],
        actions[:, 0, 1] == 1, bm[:, 1],
        info["bomb_placed"][:, 0], (held > 0).astype(jnp.float32), item_release_legal,
        actions[:, 0, 1] == 2, item_placed, item_hit_self, item_hit_enemy,
        banana_hit, glue_hit, jnp.linalg.norm(p0_after - p0, axis=-1), before_home,
        after_home, before_stock, legacy_reward[:, 0], minimal_reward[:, 0], info["winner"],
        active, done,
        death_owner, trap_owner,
    ]
    return jnp.stack([jnp.asarray(value, jnp.float32) for value in values], -1)


def source_model_rows() -> dict:
    rows = {}
    for name, spec in MODELS.items():
        state_path = R / spec["state"]
        checkpoint = R / spec["checkpoint"]
        committed = json.loads((checkpoint / "COMMITTED.json").read_text())
        if sha(state_path) != committed["state_sha256"]:
            raise ValueError(f"state checkpoint hash mismatch: {name}")
        policy_path = checkpoint / "policy.pt"
        if sha(policy_path) != committed["policy_sha256"]:
            raise ValueError(f"policy checkpoint hash mismatch: {name}")
        with state_path.open("rb") as stream:
            state = pickle.load(stream)
        if not all(np.isfinite(np.asarray(leaf)).all()
                   for leaf in jax.tree.leaves(state["params"])
                   if np.asarray(leaf).dtype.kind in "fc"):
            raise ValueError(f"nonfinite params: {name}")
        rows[name] = dict(
            state=spec["state"], checkpoint=spec["checkpoint"],
            state_sha256=sha(state_path), policy_sha256=sha(policy_path),
            run_seed=spec["run_seed"], update=int(state["update"]),
            arm=(name.split('_s')[0] if not name.startswith('parent_')
                 else 'parent'), state_lineage_arm=state.get("arm"),
        )
    return rows


def write_manifest() -> dict:
    sources = source_model_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema="qqt.expert-fullmatch-reward-alignment/v3",
        id="EXPERT-FULLMATCH-REWARD-ALIGNMENT-20261026-A2",
        namespace=OUT.relative_to(R).as_posix(),
        purpose="natural full-match reward-alignment; fixed outcomes are not a skill promotion gate",
        checkpoints=sources,
        engine="native JAX JE07 transition with production JE10 training reward path",
        opponent="fixed JAX rule3; not production JS Hunter",
        teacher_off=True,
        natural_spawn_only=True,
        synthetic_carry_or_bank_initialization=False,
        action_modes=["categorical", "greedy"],
        games_per_checkpoint_mode=GAMES_PER_MODE,
        seed_start=SEED_START,
        seed_end=SEED_START + GAMES_PER_MODE - 1,
        paired_seed_list=list(range(SEED_START, SEED_START + GAMES_PER_MODE)),
        initial_conditions="same env-init seed and episode index across checkpoints and action modes",
        actor_rng="independent per seed; same seed stream across checkpoint and mode rows",
        opponent_rng="independent from actor RNG; fixed rule3",
        environment_rng="independent from actor/opponent RNG; same seed stream across rows",
        horizon_ticks=2400,
        settlement_tail_max_ticks=MAX_TICKS - 2400,
        scan_chunk_ticks=CHUNK,
        scan_chunk_count=MAX_TICKS // CHUNK,
        gamma=GAMMA,
        reward_scoring=dict(
            legacy="native engine output, including transport pending settlement/clawback",
            minimal="frozen D1 select_reward adapter scored on the same transition trace",
            no_cross_reward_scale_skill_ranking=True,
            exclusive_legacy_terms=list(EXCLUSIVE_LEGACY_TERMS),
            reported_total_components=list(REPORTED_TOTAL_COMPONENTS),
            exclusive_minimal_terms=list(EXCLUSIVE_MINIMAL_TERMS),
            per_transition_decomposition_tolerance=DECOMPOSITION_ATOL,
            metrics_and_counts_not_added_to_reward_totals=True,
        ),
        termination=dict(
            distinguish_capture_native_game_end_timeout_and_artificial_truncation=True,
            no_value_bootstrap=True,
            per_episode_completion_required=True,
        ),
        completion="32 seed-matched games per checkpoint per action mode; every seed retained, including no-event games",
        incomplete_D_coverage=("D has only the first completed paired seed at this point; the current minimal checkpoint is u064 and the legacy checkpoint is u320. These rows are diagnostic and are not a paired endpoint contrast."),
        CPU_only=True,
        no_training_replay=True,
        no_GPU_control=True,
        no_holdout_tuning=True,
        created_unix=time.time(),
        split='diagnostic_dev; not independent holdout', seat=0,
        game_seed_semantics='seed labels are SEED_START+episode_index; actual keys use fold_in(PRNGKey(SEED_START), episode_index*13+offset), offsets env=1 actor=5',
        map='natural initialized layout0; initial wall/brick fingerprint recorded per batch',
        source_sha256={name: sha(R/name) for name in (
            'scripts/passive_evaluation_metrics.py',
            'scripts/expert_reward_alignment_fullmatch.py',
            'scripts/carrier_mechanism_engine.py', 'scripts/combat_item_engine.py',
            'scripts/combat_item_events.py', 'scripts/expert_reward_ablation.py',
            'scripts/expert_reward_ablation_prepare.py',
            'scripts/je10_ops.py', 'scripts/je10_control_credit.py',
            'scripts/je09_route_pbrs.py', 'scripts/je08_bun_potential.py',
            'scripts/je09_bun_scenes.py', 'jax_bomb/je07_bun_env.py',
            'jax_bomb/je05_core_env.py', 'jax_bomb/bun_jax_bots.py')},
    )
    atomic_json(MANIFEST, manifest)
    return manifest


def select_tree(mask, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b),
        new, old)


def choose(params, states, aux, keys, categorical):
    keys, ka, kb, ko, ke = jnp.swapaxes(
        jax.vmap(lambda key: jax.random.split(key, 5))(keys), 0, 1)
    obs, gv, mm, bm = E.inputs(states)
    mv, ab, *_ = E.prediction(params, states)
    bm = bm.at[:, 1:].set(bm[:, 1:] & ~aux["terminal"][:, None])
    move_probs, _ = _masked_entropy(mv, mm)
    ability_probs, _ = _masked_entropy(ab, bm)
    move_logits = jnp.where(mm, mv, -1e9)
    ability_logits = jnp.where(bm, ab, -1e9)
    sampled_move = jax.vmap(jax.random.categorical)(ka, move_logits)
    sampled_ability = jax.vmap(jax.random.categorical)(kb, ability_logits)
    greedy_move = jnp.argmax(move_logits, -1)
    greedy_ability = jnp.argmax(ability_logits, -1)
    own = jnp.stack((jnp.where(categorical, sampled_move, greedy_move),
                     jnp.where(categorical, sampled_ability, greedy_ability)), -1)
    allmm, allbm = jax.vmap(E.env.legal_mask)(states)
    enemybm = allbm[:, 1].at[:, 1:].set(allbm[:, 1, 1:] & ~aux["terminal"][:, None])

    def rule(state, move_mask, ability_mask, key):
        one = jax.tree.map(lambda value: value[None], state)
        return E.bun_jax_bots.rule_bot_actions(
            one, jnp.array([1]), move_mask[None], ability_mask[None], key,
            jnp.array([3]))[0]

    enemy = jax.vmap(rule)(states, allmm[:, 1], enemybm, ko)
    actions = jnp.stack((own, enemy), axis=1)
    return keys, ke, actions, (move_probs, ability_probs)


def make_scan():
    names = prep.reward_component_names(E.ops.COMPONENTS, item_events.METRICS)

    def scan_chunk(params, states, aux, keys, ended, categorical):
        def body(carry, _):
            st, ax, key, stopped = carry
            next_key, env_key, actions, policy = choose(params, st, ax, key, categorical)
            after, next_aux, legacy_reward, done, info, parts = T.transition(
                st, ax, actions, env_key, arm="control")
            kill = control_credit.parent_reward_events(st, info)["surviving_kill"]
            minimal_reward, minimal_terms, minimal_parts = reward_adapter.select_reward(
                "minimal", legacy_reward, info, parts,
                prep.reward_component_names(E.ops.COMPONENTS, item_events.METRICS),
                kill_event=kill)
            active = ~stopped
            after = select_tree(stopped, st, after)
            next_aux = select_tree(stopped, ax, next_aux)
            next_key = jnp.where(stopped[:, None], key, next_key)
            event_rows = []
            for event_name in EVENT_NAMES:
                event = jnp.asarray(info.get(event_name, jnp.zeros_like(done)))
                event_rows.append(event[:, 0].astype(jnp.float32))
            events = jnp.stack(event_rows, -1)
            _, _, move_mask, ability_mask = E.inputs(st)
            telemetry = _passive_snapshot(
                st, after, info, actions, policy, move_mask, ability_mask,
                legacy_reward, minimal_reward, active, done)
            output = (
                jnp.where(active, legacy_reward[:, 0], 0.),
                jnp.where(active, minimal_reward[:, 0], 0.),
                jnp.where(active[:, None], parts[:, 0], 0.),
                jnp.where(active[:, None], minimal_parts[:, 0], 0.),
                jnp.where(active[:, None], events, 0.),
                jnp.where(active, done, False),
                active,
                jnp.where(active, after.t, 0),
                jnp.where(active[:, None], telemetry, 0.),
            )
            return (after, next_aux, next_key, stopped | done), output

        return jax.lax.scan(body, (states, aux, keys, ended), None, length=CHUNK)

    return jax.jit(scan_chunk), names


def initial_batch(episode_indices):
    env_keys = jax.vmap(lambda idx: jax.random.fold_in(
        jax.random.PRNGKey(SEED_START), idx * 13 + 1))(episode_indices)
    actor_keys = jax.vmap(lambda idx: jax.random.fold_in(
        jax.random.PRNGKey(SEED_START), idx * 13 + 5))(episode_indices)
    states, aux = E.initial(env_keys, jnp.array(0, jnp.int32))
    aux = dict(aux, tactics=item_events.state(len(episode_indices)),
               provenance_source=jnp.full((len(episode_indices),), -1, jnp.int32))
    return states, aux, actor_keys


def chunk_sizes(total=MAX_TICKS, chunk=CHUNK):
    if total <= 0 or chunk <= 0 or total % chunk:
        raise ValueError("full-match tick budget must be a positive multiple of scan chunk")
    return [chunk] * (total // chunk)


def sum_discounted(values):
    result = np.zeros(values.shape[1], np.float64)
    for t in range(values.shape[0] - 1, -1, -1):
        result = values[t].astype(np.float64) + GAMMA * result
    return result.tolist()


def per_transition_decomposition_error(parts, reward, names, term_names):
    total_index = names.index("combined_total")
    exclusive_indices = [names.index(name) for name in term_names]
    total = np.asarray(parts[..., total_index], dtype=np.float64)
    recomposed = np.asarray(parts[..., exclusive_indices], dtype=np.float64).sum(-1)
    reward = np.asarray(reward, dtype=np.float64)
    return dict(
        max_abs_exclusive_to_total=float(np.max(np.abs(recomposed-total), initial=0.0)),
        max_abs_total_component_to_reward=float(np.max(np.abs(total-reward), initial=0.0)),
    )


def native_winner(trace_rows):
    """The first native terminal result owns the match, before residual hazards."""
    return next((int(row['winner']) for row in trace_rows
                 if row.get('winner') in (-1, 0, 1)), None)


def completed_game_count(condition_rows, active_games=0):
    return sum(row['games'] for row in condition_rows) + active_games


def write_batch_trace(output, model, mode, indices, arrays, names):
    """Persist numeric observations for offline recomputation, with no pickle."""
    folder = Path(output) / 'traces'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{model}_{mode}_{indices[0]:04d}.npz'
    keys = ('legacy_reward','minimal_reward','legacy_parts','minimal_parts',
            'events','done','active','ticks','telemetry')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **dict(zip(keys, arrays)),
                            telemetry_names=np.asarray(TELEMETRY_NAMES),
                            component_names=np.asarray(names),
                            event_names=np.asarray(EVENT_NAMES),
                            episode_indices=np.asarray(indices))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def summarize_episodes(model_name, mode, episode_indices, arrays, final_states, final_aux, names):
    legacy, minimal, legacy_parts, minimal_parts, events, done, active, ticks, telemetry = arrays
    legacy = np.asarray(legacy); minimal = np.asarray(minimal)
    legacy_parts = np.asarray(legacy_parts); minimal_parts = np.asarray(minimal_parts)
    events = np.asarray(events); done = np.asarray(done); active = np.asarray(active)
    ticks = np.asarray(ticks)
    telemetry = np.asarray(telemetry)
    results = []
    for col, episode in enumerate(episode_indices):
        valid = active[:, col].astype(bool)
        lret = float(legacy[valid, col].sum())
        mret = float(minimal[valid, col].sum())
        le_terms = legacy_parts[valid, col].sum(0)
        mi_terms = minimal_parts[valid, col].sum(0)
        event_sum = events[valid, col].sum(0)
        legacy_decomposition = per_transition_decomposition_error(
            legacy_parts[valid, col], legacy[valid, col], names,
            EXCLUSIVE_LEGACY_TERMS)
        minimal_decomposition = per_transition_decomposition_error(
            minimal_parts[valid, col], minimal[valid, col], names,
            tuple(name for name in ("transport_capture", "settled_kill", "global_safety",
                                    "control_entry_reward", "control_residual_reward",
                                    "route_shaped_reward") if name in names))
        if (legacy_decomposition["max_abs_exclusive_to_total"] > DECOMPOSITION_ATOL or
                legacy_decomposition["max_abs_total_component_to_reward"] > DECOMPOSITION_ATOL):
            raise ValueError(
                "legacy exclusive reward decomposition mismatch: "
                f"{model_name}/{episode}; "
                f"exclusive_to_total={legacy_decomposition['max_abs_exclusive_to_total']:.9g}; "
                f"total_to_reward={legacy_decomposition['max_abs_total_component_to_reward']:.9g}")
        if minimal_decomposition["max_abs_exclusive_to_total"] > DECOMPOSITION_ATOL or \
                minimal_decomposition["max_abs_total_component_to_reward"] > DECOMPOSITION_ATOL:
            raise ValueError(
                "minimal exclusive reward decomposition mismatch: "
                f"{model_name}/{episode}; "
                f"exclusive_to_total={minimal_decomposition['max_abs_exclusive_to_total']:.9g}; "
                f"total_to_reward={minimal_decomposition['max_abs_total_component_to_reward']:.9g}")
        term_ticks = np.flatnonzero(done[:, col])
        terminal = bool(term_ticks.size)
        final_tick = int(term_ticks[0] + 1) if terminal else int(valid.sum())
        horizon_hit = terminal and int(ticks[term_ticks[0], col]) >= 2400
        if not terminal:
            termination = "artificial_truncation"
        elif event_sum[0] > 0:
            termination = "native_capture"
        elif horizon_hit:
            termination = "game_timeout"
        else:
            termination = "native_game_end"
        ledger = {
            "transport_balance_p0": float(final_aux["transport"]["balance"][col, 0]),
            "transport_pending_capture_p0": float(final_aux["transport"]["pending_capture"][col, 0]),
            "transport_episode_gross_p0": float(final_aux["transport"]["episode_gross"][col, 0]),
            "control_balance_p0": float(final_aux["control_credit"]["balance"][col, 0]),
            "control_gross_p0": float(final_aux["control_credit"]["gross"][col, 0]),
            "route_phi_p0": float(final_aux["route_pbrs"]["phi"][col, 0]),
            "route_offset_p0": float(final_aux["route_pbrs"]["offset"][col, 0]),
        }
        reward_ledgers = reward_ledger_snapshot(final_aux, col)
        trace_rows = []
        for values in telemetry[valid, col]:
            row = {name: float(value) for name, value in zip(TELEMETRY_NAMES, values)}
            phase_code = int(row.pop("phase_code"))
            row["phase"] = passive_metrics.PHASES[phase_code]
            for key in (
                "enemy_alive_before", "enemy_alive_after", "enemy_death",
                "enemy_native_trap_enter", "own_capture", "own_return_bun",
                "own_death", "own_drop", "stock_pickup", "loose_enemy_pickup",
                "loose_own_pickup", "native_transfer", "bomb_request", "bomb_legal",
                "bomb_placed", "item_release_opportunity", "item_release_request",
                "item_placed", "item_hit_self", "item_hit_enemy", "banana_hit_proxy",
                "glue_hit_proxy", "active", "done",
            ):
                row[key] = bool(row[key])
            for key in (
                "own_carried_before", "own_carried_after", "enemy_trapped_before",
                "enemy_respawn_before", "own_trapped_after", "own_move", "own_ability",
                "item_inventory", "window_cause_owner", "winner", "tick",
            ):
                row[key] = int(row[key])
            trace_rows.append(row)
        passive = passive_metrics.summarize_game(
            f"{model_name}:{mode}:{int(episode)}", trace_rows,
            termination=termination,
            winner=native_winner(trace_rows),
            model=model_name, action_mode=mode)
        results.append(dict(
            episode_index=int(episode), seed=SEED_START + int(episode),
            valid_transitions=int(valid.sum()), terminal=terminal,
            termination=termination, settlement_tail_ticks=max(0, final_tick - 2400),
            artificial_truncation=not terminal,
            undiscounted_legacy_return=lret,
            discounted_legacy_return_gamma_0_995=sum_discounted(legacy[:, col:col+1])[0],
            undiscounted_minimal_return=mret,
            discounted_minimal_return_gamma_0_995=sum_discounted(minimal[:, col:col+1])[0],
            legacy_reward_terms={name: float(le_terms[names.index(name)])
                                 for name in EXCLUSIVE_LEGACY_TERMS if name in names},
            legacy_total_component=float(le_terms[names.index("combined_total")]),
            legacy_decomposition=legacy_decomposition,
            minimal_reward_terms={
                "delivery": float(mi_terms[names.index("transport_capture")]),
                "kill": float(mi_terms[names.index("settled_kill")]),
                "death": float(mi_terms[names.index("global_safety")]),
                "control_entry": float(mi_terms[names.index("control_entry_reward")]),
                "control_residual": float(mi_terms[names.index("control_residual_reward")]),
                "route_pbrs": float(mi_terms[names.index("route_shaped_reward")]),
                "combined_total": float(mi_terms[names.index("combined_total")]),
            },
            minimal_total_component=float(mi_terms[names.index("combined_total")]),
            minimal_decomposition=minimal_decomposition,
            events={name: int(value) for name, value in zip(EVENT_NAMES, event_sum)},
            native_event_winner=native_winner(trace_rows),
            transport_ledger=ledger,
            reward_ledger_snapshot=reward_ledgers,
            passive_metrics=passive,
        ))
    return results


def reward_ledger_snapshot(aux, environment_index):
    """Keep final balances/pending flags from each real reward ledger."""
    fields = {
        "balance", "gross", "episode_gross", "pending", "pending_capture",
        "captured", "event_id", "owner", "last_tick", "residual_paid",
        "phi", "offset", "phase", "identity", "retired", "paid", "ticks",
        "start_distance", "start_home", "nonce", "credit", "cycle_paid",
        "claimed_nonce", "slide", "slide_distance", "slide_valid",
    }
    roots = ("transport", "control_credit", "drop", "bridge", "potential",
             "route_pbrs", "carry_items")
    result = {}

    def collect(node, batch_size):
        values = {}
        if not isinstance(node, dict):
            return values
        for name, value in node.items():
            if isinstance(value, dict):
                child = collect(value, batch_size)
                if child:
                    values[name] = child
                continue
            if name not in fields:
                continue
            array = np.asarray(value)
            if array.ndim and array.shape[0] == batch_size:
                selected = array[environment_index]
                values[name] = selected.tolist() if selected.ndim else selected.item()
        return values

    batch_size = int(np.asarray(aux["terminal"]).shape[0])
    for root in roots:
        if root in aux:
            snapshot = collect(aux[root], batch_size)
            if snapshot:
                result[root] = snapshot
    return result


def run_batch(params, model_name, mode, episode_indices, scan, names):
    states, aux, keys = initial_batch(jnp.asarray(episode_indices, jnp.int32))
    map_fingerprints = [hashlib.sha256(
        np.asarray(states.wall[i]).tobytes()+np.asarray(states.brick[i]).tobytes()
    ).hexdigest() for i in range(len(episode_indices))]
    ended = jnp.zeros((len(episode_indices),), bool)
    outputs = []
    for _ in chunk_sizes():
        if bool(np.asarray(jnp.all(ended))):
            break
        (states, aux, keys, ended), out = scan(
            params, states, aux, keys, ended, mode == "categorical")
        jax.block_until_ready(states.t)
        outputs.append(tuple(np.asarray(x) for x in out))
    merged = tuple(np.concatenate([part[i] for part in outputs], axis=0)
                   for i in range(len(outputs[0])))
    trace = write_batch_trace(OUT, model_name, mode, episode_indices, merged, names)
    results = summarize_episodes(model_name, mode, episode_indices, merged,
                                 states, aux, names)
    for i, row in enumerate(results):
        row['provenance'] = dict(
            engine='native JAX JE07', opponent='fixed JAX rule3', seat=0,
            split='diagnostic_dev', natural=True, teacher=False,
            initial_wall_brick_sha256=map_fingerprints[i],
            trace=trace.relative_to(R).as_posix(), trace_sha256=sha(trace),
            telemetry_column=i, state_sha256=sha(R/MODELS[model_name]['state']),
            policy_sha256=sha(R/MODELS[model_name]['checkpoint']/'policy.pt'))
    return results


def load_params(spec):
    with (R / spec["state"]).open("rb") as stream:
        return pickle.load(stream)["params"]


def paired_game_comparison(rows):
    by_model_mode = {(row["model"], row["action_mode"]): row for row in rows}
    parent = "parent_r1_s2751_u064"
    candidate = "minimal_s2851_u064"
    if (parent, "categorical") not in by_model_mode or (candidate, "categorical") not in by_model_mode:
        return {"status": "not_a_complete_declared_pair", "comparisons": []}
    comparisons = []
    for mode in ("categorical", "greedy"):
        a = by_model_mode[(parent, mode)]
        b = by_model_mode[(candidate, mode)]
        amap = {ep["seed"]: ep["passive_metrics"] for ep in a["episodes"]}
        bmap = {ep["seed"]: ep["passive_metrics"] for ep in b["episodes"]}
        common = sorted(set(amap) & set(bmap))
        pair_rows = []
        for seed in common:
            pa = amap[seed]["full_game"]
            pb = bmap[seed]["full_game"]
            pair_rows.append({
                "game_seed": seed,
                "parent_delivery_events": pa["delivery_events"]["count"],
                "candidate_delivery_events": pb["delivery_events"]["count"],
                "parent_own_deaths": pa["own_deaths"]["count"],
                "candidate_own_deaths": pb["own_deaths"]["count"],
                "parent_winner": pa["net_wins"]["numerator"] if pa["net_wins"] else None,
                "candidate_winner": pb["net_wins"]["numerator"] if pb["net_wins"] else None,
                "same_action_mode_and_opponent": True,
            })
        comparisons.append({
            "parent_model": parent,
            "candidate_model": candidate,
            "action_mode": mode,
            "paired_games": len(pair_rows),
            "paired_game_seed_rows": pair_rows,
            "mean_delivery_event_delta_candidate_minus_parent": (
                float(np.mean([p["candidate_delivery_events"] - p["parent_delivery_events"]
                               for p in pair_rows])) if pair_rows else None),
            "mean_own_death_delta_candidate_minus_parent": (
                float(np.mean([p["candidate_own_deaths"] - p["parent_own_deaths"]
                               for p in pair_rows])) if pair_rows else None),
            "inferential_claim": None,
            "training_seed_variance_estimated": False,
        })
    return {"status": "matched_native_jax_diagnostic", "comparisons": comparisons}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    owner_lock = lock(OUT / "owner.lock")
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else write_manifest()
    for name, expected in manifest.get('source_sha256', {}).items():
        if sha(R/name) != expected:
            raise ValueError(f'evaluation source differs from frozen manifest: {name}')
    T.configure()
    provenance.install_provenance()
    scan, names = make_scan()
    total_games = len(MODELS) * 2 * GAMES_PER_MODE
    results = []
    started = time.time()
    for model_name, spec in MODELS.items():
        params = load_params(spec)
        for mode in ("categorical", "greedy"):
            mode_rows = []
            for offset in range(0, GAMES_PER_MODE, BATCH_SIZE):
                episodes = list(range(offset, min(offset + BATCH_SIZE, GAMES_PER_MODE)))
                mode_rows.extend(run_batch(params, model_name, mode, episodes, scan, names))
                atomic_json(OUT / "PROGRESS.json", dict(
                    completed_games=completed_game_count(results, len(mode_rows)), total_games=total_games,
                    current_model=model_name, action_mode=mode,
                    current_mode_completed=len(mode_rows), cpu_only=True,
                    no_training_replay=True, unix=time.time()))
            row = dict(model=model_name, action_mode=mode, games=len(mode_rows),
                       seed_start=SEED_START, seed_end=SEED_START + GAMES_PER_MODE - 1,
                       termination_counts={}, legacy={}, minimal={}, episodes=mode_rows)
            for ep in mode_rows:
                row["termination_counts"][ep["termination"]] = row["termination_counts"].get(ep["termination"], 0) + 1
            for arm in ("legacy", "minimal"):
                u = [ep[f"undiscounted_{arm}_return"] for ep in mode_rows]
                d = [ep[f"discounted_{arm}_return_gamma_0_995"] for ep in mode_rows]
                row[arm] = dict(
                    mean_undiscounted=float(np.mean(u)),
                    mean_discounted_gamma_0_995=float(np.mean(d)),
                    episode_returns_undiscounted=u,
                    episode_returns_discounted_gamma_0_995=d,
                    exclusive_component_sum={name: float(sum(ep[f"{arm}_reward_terms"].get(name, 0.)
                                                               for ep in mode_rows))
                                             for name in (EXCLUSIVE_LEGACY_TERMS if arm == "legacy"
                                                          else EXCLUSIVE_MINIMAL_TERMS)},
                    reported_total_component_sum=float(sum(
                        ep[f"{arm}_total_component"] for ep in mode_rows)),
                    max_abs_decomposition_error=max(
                        ep[f"{arm}_decomposition"]["max_abs_exclusive_to_total"]
                        for ep in mode_rows),
                    events={name: int(sum(ep["events"][name] for ep in mode_rows))
                            for name in EVENT_NAMES},
                    episodes_with_native_capture=sum(ep["events"]["capture"] > 0 for ep in mode_rows),
                    episodes_with_death=sum(ep["events"]["death"] > 0 for ep in mode_rows),
                )
            results.append(row)
            row['passive_cohort'] = passive_metrics.summarize_cohort(
                [episode['passive_metrics'] for episode in mode_rows])
            atomic_json(OUT / f"{model_name}_{mode}.json", dict(
                schema="qqt.expert-fullmatch-reward-alignment-row/v2",
                manifest=MANIFEST.relative_to(R).as_posix(), row=row,
                source_sha256={"manifest": sha(MANIFEST)},
                no_cross_scale_skill_ranking=True, cpu_only=True, unix=time.time()))
    summary = dict(
        schema="qqt.expert-fullmatch-reward-alignment-summary/v2",
        manifest=MANIFEST.relative_to(R).as_posix(),
        manifest_sha256=sha(MANIFEST),
        games_per_checkpoint_mode=GAMES_PER_MODE,
        total_games=total_games,
        rows=results,
        paired_parent_candidate=paired_game_comparison(results),
        passive_metric_coverage=passive_metrics.coverage_matrix(),
        caveats=[
            "Native JAX training transition and fixed JAX rule3 opponent; this is not production JS Hunter.",
            "Per-checkpoint/mode sample size is fixed at 32 complete natural matches; do not treat games as independent training seeds.",
            "Legacy and minimal reward totals have different scales and cannot alone rank policy skill.",
            "No artificial carrying or Bank state initialization; all episodes start from natural spawn.",
            "Every registered seed remains in output, including no-event and artificial-truncation rows.",
        ],
        source_sha256={"manifest": sha(MANIFEST)},
        elapsed_seconds=time.time() - started,
        cpu_only=True, no_training_replay=True, no_GPU_control=True,
        completed_unix=time.time())
    for name, expected in manifest.get('source_sha256', {}).items():
        if sha(R/name) != expected:
            raise ValueError(f'evaluation source changed while running: {name}')
    atomic_json(OUT / "SUMMARY.json", summary)
    atomic_json(OUT / "COMPLETE.json", dict(
        passed=True, total_games=total_games, all_checkpoint_mode_rows=len(results),
        artificial_truncations=sum(ep["termination_counts"].get("artificial_truncation", 0)
                                   for ep in results),
        manifest_sha256=sha(MANIFEST), summary_sha256=sha(OUT / "SUMMARY.json"),
        cpu_only=True, unix=time.time()))
    print(json.dumps({"complete": True, "total_games": total_games,
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "summary_sha256": sha(OUT / "SUMMARY.json")}, indent=2))


if __name__ == "__main__":
    main()
