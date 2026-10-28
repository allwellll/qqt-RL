"""PPO-only continuation worker for the fixed R2 long-horizon budget."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
import optax

from scripts import ppo_scale_longrun_train as P
from scripts.carrier_mechanism_analysis import bank, merged
from scripts.carrier_mechanism_stats import validate_bank_record
from scripts.population_run_state import atomic_json, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/ppo_scale_20261025/r2_longrun"
R1 = R / "runs/ppo_scale_20261025/r1_longrun"
GLOBAL_PAUSE = R / "runs/retention_20261021/USER_GPU_PAUSE.json"
ARM = "ppo_scaled_batch_continuation"
SOURCE_ARM = "ppo_scaled_batch"
OLD_STEPS_PER_SEED = 1_048_576
ENV_STEPS_PER_UPDATE = 16 * 128 * 8
OPTIMIZER_STEPS_PER_UPDATE = 32
SAVE_EVERY = 64


def read(path):
    return json.loads(Path(path).read_text())


def notify(message):
    print(message, flush=True)
    atomic_json(B / "notifications" / f"{time.time_ns()}.json",
                {"message": message, "unix": time.time()})


def authorize():
    if (B / "USER_STOP.json").exists():
        return False
    protocol = read(B / "PROTOCOL.json")
    authorization = read(B / "SCOPED_GPU_AUTHORIZATION.json")
    if sha(GLOBAL_PAUSE) != authorization["global_pause_sha256"]:
        return False
    assert protocol["id"] == "PPO-SCALE-LONGRUN-20261025-R2"
    assert authorization["authorized_by"] == "explicit_user_instruction"
    assert authorization["protocol_sha256"] == sha(B / "PROTOCOL.json")
    assert protocol["budget"]["new_environment_steps_per_seed"] == 33_554_432
    assert protocol["common"]["dagger_coefficient"] == 0.0
    assert protocol["common"]["combat_kl_coefficient"] == 0.0
    assert protocol["parent"]["namespace"] == str(R1.relative_to(R))
    assert protocol["old_queues_remain_paused"] is True
    return True


def bind_new_namespace():
    # Reuse the exact frozen R1 PPO collector/updater implementation while
    # redirecting all paths and authorization to this new namespace.
    P.B, P.R, P._read, P.authorize, P.notify = B, R, read, authorize, notify
    P.C.B, P.C.R, P.C.read, P.C.authorize, P.C.notify = B, R, read, authorize, notify
    P.T.B = B


def setup():
    bind_new_namespace()
    assert authorize()
    seal = read(B / "SOURCE_SEAL.json")["sources"]
    for name, expected in seal.items():
        assert sha(R / name) == expected, f"sealed source changed: {name}"
    P.T.configure()
    P.C.install_provenance()
    assert jax.default_backend() == "gpu" and jax.local_device_count() == 8
    items = [x for x in bank() if x["record"]["split"] == "train"]
    assert len(items) == 32
    for item in items:
        validate_bank_record(item["record"])
    bs, ba = merged(items, "states"), merged(items, "aux")
    return items, bs, ba


def source_for_seed(seed):
    info = read(B / "SOURCE_MANIFEST.json")["sources"][str(seed)]
    cp = R / info["checkpoint"]
    receipt = read(cp / "COMMITTED.json")
    assert receipt["state_sha256"] == info["state_sha256"] == sha(cp / "state.pkl")
    assert receipt["policy_sha256"] == info["policy_sha256"] == sha(cp / "policy.pt")
    with (cp / "state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    assert (state["seed"], state["arm"], state["update"]) == (seed, SOURCE_ARM, 64)
    assert set(state) >= {"params", "adam", "states", "aux", "keys", "history"}
    assert np.asarray(state["keys"]).shape == (8, 16, 2)
    assert all(np.isfinite(np.asarray(v)).all()
               for key in ("params", "adam", "states", "aux", "keys", "history")
               for v in jax.tree.leaves(state[key])
               if np.asarray(v).dtype.kind in "fc")
    return state, cp, info


def finite_tree(value):
    return all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(value)
               if np.asarray(v).dtype.kind in "fc")


def action_distribution(batch):
    actions = np.asarray(batch[2]).reshape(-1, np.asarray(batch[2]).shape[-1])
    movement = np.bincount(actions[:, 0].astype(np.int64), minlength=5)
    ability = np.bincount(actions[:, 1].astype(np.int64), minlength=3)
    return dict(
        samples=int(len(actions)),
        movement={str(i): int(v) for i, v in enumerate(movement)},
        ability={str(i): int(v) for i, v in enumerate(ability)},
    )


def model_deltas(before, after):
    return P.norms(before, after)


def build_optimizer():
    protocol = read(B / "PROTOCOL.json")
    return optax.chain(optax.clip_by_global_norm(.5),
                       optax.adam(protocol["common"]["lr"], eps=1e-8))


def optimizer_count(adam):
    counts = [int(np.asarray(v)) for v in jax.tree.leaves(adam)
              if np.asarray(v).shape == () and np.asarray(v).dtype.kind in "iu"]
    return counts


def checkpoint_due(new_update, total_updates=2048, stop_requested=False):
    return new_update % SAVE_EVERY == 0 or new_update == total_updates or stop_requested


def checkpoint(base, state, metrics, *, formal=True):
    new_update = int(state["new_update"])
    global_update = int(state["update"])
    cp = base / f"u{new_update:06d}"
    cp.mkdir(parents=True, exist_ok=True)
    P.C.save(cp / "state.pkl", state)
    P.C.save(cp / "policy.pt", state["params"]["actor"])
    record = dict(
        passed=True,
        arm=ARM,
        source_arm=SOURCE_ARM,
        seed=state["seed"],
        update=global_update,
        new_update=new_update,
        source_update=64,
        state_sha256=sha(cp / "state.pkl"),
        policy_sha256=sha(cp / "policy.pt"),
        metrics=metrics,
        environment_steps=new_update * ENV_STEPS_PER_UPDATE,
        lineage_environment_steps=1_048_576 + new_update * ENV_STEPS_PER_UPDATE,
        optimizer_steps=OPTIMIZER_STEPS_PER_UPDATE,
        new_optimizer_steps=new_update * OPTIMIZER_STEPS_PER_UPDATE,
        lineage_optimizer_steps=2048 + new_update * OPTIMIZER_STEPS_PER_UPDATE,
        adam_scalar_counters=optimizer_count(state["adam"]),
    )
    atomic_json(cp / "COMMITTED.json", record)
    if formal:
        full_eval = (new_update % 512 == 0 or
                     new_update == read(B / "PROTOCOL.json")["updates"])
        request = dict(
            checkpoint=str(cp.relative_to(R)),
            checkpoint_state_sha256=record["state_sha256"],
            checkpoint_policy_sha256=record["policy_sha256"],
            seed=state["seed"],
            source_update=64,
            new_update=new_update,
            environment_steps=record["environment_steps"],
            level="full" if full_eval else "light",
            holdout=new_update == read(B / "PROTOCOL.json")["updates"],
            controls_gpu=False,
        )
        atomic_json(B / "eval_queue" /
                    f"s{state['seed']}_u{new_update:06d}.json", request)
    return cp, record


def make_metrics(seed, new_update, batch, diag, loss, deltas,
                 elapsed_seconds, rollout_seconds, history):
    parts, flags, opportunities, rewards, values, advantages, origins = map(np.asarray, diag)
    flat_flags = flags.reshape((-1, flags.shape[-1]))
    flat_opp = opportunities.reshape((-1, opportunities.shape[-1]))
    flat_origin = origins.reshape(-1)
    loss_values = np.asarray(loss).mean((0, 1))
    optimizer_steps = OPTIMIZER_STEPS_PER_UPDATE
    env_steps = new_update * ENV_STEPS_PER_UPDATE
    result = dict(
        arm=ARM,
        seed=seed,
        source_update=64,
        update=64 + new_update,
        new_update=new_update,
        finite=True,
        environment_steps=env_steps,
        lineage_environment_steps=1_048_576 + env_steps,
        optimizer_steps=optimizer_steps,
        new_optimizer_steps=new_update * optimizer_steps,
        lineage_optimizer_steps=2048 + new_update * optimizer_steps,
        rollout_batch=16_384,
        global_minibatch=512,
        update_l2=deltas,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=ENV_STEPS_PER_UPDATE / max(elapsed_seconds, 1e-9),
        iteration_seconds=elapsed_seconds,
        rollout_seconds=rollout_seconds,
        startup_compile_included=(new_update == 1),
        loss=loss_values.tolist(),
        loss_fields=["total", "ppo_policy", "value", "entropy", "approx_kl",
                     "clip_fraction", "gradient_norm", "ability_gradient_norm"],
        actor_ability_update_l2=float(deltas.get("ability", 0.0)),
        action_distribution=action_distribution(batch),
        ability_sampled=int(flat_flags[:, 3].sum()),
        greedy_release=int(flat_flags[:, 4].sum()),
        release_probability_sum=float(flat_flags[:, 5].sum()),
        item_release_opportunities=int(flat_opp[:, 2].sum()),
        item_contact_labels=int(flat_opp[:, 3].sum()),
        carry_inventory_ticks=int(flat_flags[:, 1].sum()),
        carry_ticks=int(flat_flags[:, 0].sum()),
        deaths=int(flat_flags[:, 8].sum()),
        source_exposure=dict(
            natural_ticks=int((flat_origin < 0).sum()),
            real_pickup_bank_ticks=int((flat_origin >= 0).sum()),
        ),
        reward_components=np.sum(parts, axis=tuple(range(parts.ndim - 1))).tolist(),
        gae_mean=float(advantages.mean()),
        gae_variance=float(advantages.var()),
        value_explained_variance_collected=False,
        dagger_coefficient=0.0,
        combat_kl_coefficient=0.0,
        ppo_batch_source="fresh categorical learner stream only",
        auxiliary_advantages_discarded=True,
        actual_opponent_exposure=dict(current=0, ring=0, fixed=ENV_STEPS_PER_UPDATE,
                                      identity="JAX rule3"),
        checkpointed=False,
    )
    return result


def train_seed(seed, items, bank_states, bank_aux):
    protocol = read(B / "PROTOCOL.json")
    base = B / "formal" / f"{ARM}_s{seed}"
    base.mkdir(parents=True, exist_ok=True)
    source, source_cp, source_manifest = source_for_seed(seed)
    params, adam = P.rep(source["params"]), P.rep(source["adam"])
    states, aux, keys, history = source["states"], source["aux"], source["keys"], source["history"]
    start = 0
    committed = sorted(base.glob("u*/COMMITTED.json"),
                       key=lambda p: int(read(p)["new_update"]))
    if committed:
        cp = committed[-1].parent
        receipt = read(cp / "COMMITTED.json")
        assert sha(cp / "state.pkl") == receipt["state_sha256"]
        with (cp / "state.pkl").open("rb") as stream:
            restored = pickle.load(stream)
        assert finite_tree(restored)
        start = int(restored["new_update"])
        assert restored["update"] == 64 + start and restored["seed"] == seed
        params, adam = P.rep(restored["params"]), P.rep(restored["adam"])
        states, aux, keys, history = (restored["states"], restored["aux"],
                                      restored["keys"], restored["history"])
    else:
        assert sha(source_cp / "state.pkl") == source_manifest["state_sha256"]
        atomic_json(base / "FRESH_SOURCE.json", dict(
            passed=True, source_checkpoint=str(source_cp.relative_to(R)),
            source_state_sha256=source_manifest["state_sha256"],
            source_policy_sha256=source_manifest["policy_sha256"],
            params_and_Adam_loaded_exactly=True,
            env_aux_rng_history_restored=True,
            no_smoke_state_inherited=True,
        ))
    last_saved = start
    optimizer = build_optimizer()
    collect = P.C.collector(128)
    checkpoint_updates = []
    latest_state = None
    latest_row = None
    for new_update in range(start + 1, protocol["updates"] + 1):
        if not authorize():
            safe_update = new_update - 1
            if safe_update > last_saved and latest_state is not None:
                latest_row["checkpointed"] = True
                latest_row["stop_requested_at_boundary"] = True
                cp, receipt = checkpoint(base, latest_state, latest_row)
                last_saved = safe_update
                atomic_json(B / "PAUSED.json", dict(
                    passed=True, safe_boundary=True, seed=seed,
                    last_new_update=safe_update,
                    checkpoint=str(cp.relative_to(R)),
                    state_sha256=receipt["state_sha256"],
                    explicit_user_stop=True,
                ))
            else:
                atomic_json(B / "PAUSED.json", dict(
                    passed=True, safe_boundary=True, seed=seed,
                    last_new_update=start,
                    checkpoint=(str(committed[-1].parent.relative_to(R)) if committed else None),
                    source_checkpoint=str(source_cp.relative_to(R)),
                    explicit_user_stop=True,
                ))
            return False
        iteration_start = time.monotonic()
        before = P.one(params)
        (states, aux, keys), batch, diag = collect(
            params, states, aux, keys, bank_states, bank_aux)
        jax.block_until_ready(batch)
        rollout_seconds = time.monotonic() - iteration_start
        global_update = 64 + new_update
        history, _, aggregation = P.C.aggregate(batch, history, seed, global_update)
        update = P.scale_updater(64, optimizer)
        (params, adam), loss = update(
            params, adam, batch, batch,
            np.asarray(jax.random.split(jax.random.fold_in(
                jax.random.PRNGKey(seed), global_update), 8)))
        jax.block_until_ready(params)
        elapsed = time.monotonic() - iteration_start
        after = P.one(params)
        deltas = model_deltas(before, after)
        ability_delta = float(optax.global_norm(jax.tree.map(
            lambda x, y: x - y, after["actor"]["heads"]["wb"],
            before["actor"]["heads"]["wb"])))
        deltas["ability"] = ability_delta
        assert deltas["actor"] > 0 and deltas["critic"] > 0 and ability_delta > 0
        assert np.isfinite(np.asarray(loss)).all()
        assert finite_tree((after, adam, states, aux, keys))
        row = make_metrics(seed, new_update, batch, diag, loss, deltas,
                           elapsed, rollout_seconds, history)
        new_state = dict(params=after, adam=P.one(adam),
                         states=jax.device_get(states), aux=jax.device_get(aux),
                         keys=jax.device_get(keys), history=history,
                         update=global_update, new_update=new_update, seed=seed,
                         arm=SOURCE_ARM)
        latest_state, latest_row = new_state, row
        is_stop = not authorize()
        should_checkpoint = checkpoint_due(new_update, protocol["updates"], is_stop)
        cp = None
        receipt = None
        if should_checkpoint:
            row["checkpointed"] = True
            row["stop_requested_at_boundary"] = is_stop
            cp, receipt = checkpoint(base, new_state, row)
            checkpoint_updates.append(new_update)
            last_saved = new_update
            atomic_json(B / "STATUS.json", dict(
                stage="FORMAL", seed=seed, update=global_update,
                new_update=new_update, finite=True,
                environment_steps=new_update * ENV_STEPS_PER_UPDATE,
                lineage_environment_steps=1_048_576 + new_update * ENV_STEPS_PER_UPDATE,
                optimizer_steps=OPTIMIZER_STEPS_PER_UPDATE,
                cumulative_new_optimizer_steps=new_update * OPTIMIZER_STEPS_PER_UPDATE,
                steps_per_second=row["steps_per_second"],
                actor_critic_update_l2=deltas,
                checkpoint=str(cp.relative_to(R)),
                checkpoint_sha256=receipt["state_sha256"],
                unix=time.time(),
            ))
        else:
            row["checkpointed"] = False
            atomic_json(B / "STATUS.json", dict(
                stage="FORMAL", seed=seed, update=global_update,
                new_update=new_update, finite=True,
                environment_steps=new_update * ENV_STEPS_PER_UPDATE,
                optimizer_steps=OPTIMIZER_STEPS_PER_UPDATE,
                steps_per_second=row["steps_per_second"],
                last_committed_new_update=(checkpoint_updates[-1] if checkpoint_updates else start),
                unix=time.time(),
            ))
        log_path = base / "updates.jsonl"
        with log_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        params, adam = P.rep(after), P.rep(P.one(adam))
        if new_update == 1:
            notify(
                f"R2首个正式续训更新 s{seed} old-u65/new-u1 finite；"
                f"Actor Δ{deltas['actor']:.6g} Critic Δ{deltas['critic']:.6g} "
                f"ability Δ{ability_delta:.6g}，16,384 categorical步、32 optimizer步；"
                f"loss={row['loss'][:6]}。正式续训继续。"
            )
        elif should_checkpoint:
            notify(
                f"R2续训检查点 s{seed} new-u{new_update} 原子保存，"
                f"累计新增{new_update * ENV_STEPS_PER_UPDATE}步，"
                f"SHA {receipt['state_sha256'][:16]}；异步CPU/JS评估已排队。"
            )
        if is_stop:
            atomic_json(B / "PAUSED.json", dict(
                passed=True, safe_boundary=True, seed=seed,
                last_new_update=new_update, source_update=64,
                checkpoint=str(cp.relative_to(R)),
                state_sha256=receipt["state_sha256"],
                explicit_user_stop=True,
            ))
            return False

    final_cp = base / f"u{protocol['updates']:06d}"
    final_receipt = read(final_cp / "COMMITTED.json")
    assert final_receipt["new_update"] == protocol["updates"]
    atomic_json(base / "COMPLETE.json", dict(
        passed=True, seed=seed, arm=ARM, source_update=64,
        update=64 + protocol["updates"], new_update=protocol["updates"],
        environment_steps=protocol["budget"]["new_environment_steps_per_seed"],
        optimizer_steps=protocol["budget"]["new_optimizer_steps_per_seed"],
        lineage_environment_steps=OLD_STEPS_PER_SEED +
                                  protocol["budget"]["new_environment_steps_per_seed"],
        state_sha256=final_receipt["state_sha256"],
        policy_sha256=final_receipt["policy_sha256"],
        saved_checkpoint=str(final_cp.relative_to(R)),
        no_peak_selection=True,
    ))
    notify(
        f"R2固定seed终态 s{seed}: 2,048新update、33,554,432新增环境步、"
        f"65,536新增优化步，SHA {final_receipt['state_sha256'][:16]}；"
        "按冻结顺序立即接续下一seed，CPU/JS不作门禁。"
    )
    return True


def smoke():
    items, bank_states, bank_aux = setup()
    seed = read(B / "PROTOCOL.json")["paired_seeds"][0]
    source, source_cp, source_info = source_for_seed(seed)
    params, adam = P.rep(source["params"]), P.rep(source["adam"])
    optimizer = build_optimizer()
    (states, aux, keys), batch, diag = P.C.collector(8)(
        params, source["states"], source["aux"], source["keys"],
        bank_states, bank_aux)
    history, _, _ = P.C.aggregate(batch, source["history"], seed, 65)
    (next_params, next_adam), loss = P.scale_updater(64, optimizer)(
        params, adam, batch, batch,
        np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 65), 8)))
    jax.block_until_ready(next_params)
    before, after = P.one(params), P.one(next_params)
    deltas = P.norms(before, after)
    ability_delta = float(optax.global_norm(jax.tree.map(
        lambda x, y: x - y, after["actor"]["heads"]["wb"],
        before["actor"]["heads"]["wb"])))
    assert all(v > 0 for v in deltas.values()) and ability_delta > 0
    assert np.isfinite(np.asarray(loss)).all()
    smoke_state = dict(params=after, adam=P.one(next_adam),
                       states=jax.device_get(states), aux=jax.device_get(aux),
                       keys=jax.device_get(keys), history=history,
                       update=65, new_update=1, seed=seed, arm=SOURCE_ARM)
    cp, receipt = checkpoint(B / "smoke" / f"s{seed}", smoke_state,
                             dict(finite=True, actor_critic_update_l2=deltas,
                                  ability_update_l2=ability_delta,
                                  loss=np.asarray(loss).mean((0, 1)).tolist(),
                                  environment_steps=1024,
                                  optimizer_steps=2), formal=False)
    atomic_json(B / "TECHNICAL_SMOKE.json", dict(
        passed=True,
        source_checkpoint=str(source_cp.relative_to(R)),
        source_state_sha256=source_info["state_sha256"],
        restored_complete_optimizer_rng_env_aux_history=True,
        finite=True,
        actor_delta_l2=deltas["actor"],
        critic_delta_l2=deltas["critic"],
        ability_delta_l2=ability_delta,
        next_adam_counters=optimizer_count(smoke_state["adam"]),
        smoke_environment_steps=1024,
        smoke_optimizer_steps=2,
        formal_environment_steps=0,
        formal_optimizer_steps=0,
        formal_does_not_inherit_smoke=True,
        capability_gate=False,
        smoke_state_sha256=receipt["state_sha256"],
    ))
    notify(
        f"R2技术smoke通过：从seed{seed}完整u064恢复，1024环境步/2 optimizer步，"
        f"finite，Actor Δ{deltas['actor']:.6g} Critic Δ{deltas['critic']:.6g} "
        f"ability Δ{ability_delta:.6g}；formal计数仍为0。"
    )


def restore_smoke():
    items, bank_states, bank_aux = setup()
    seed = read(B / "PROTOCOL.json")["paired_seeds"][0]
    cp = B / "smoke" / f"s{seed}" / "u000001"
    receipt = read(cp / "COMMITTED.json")
    assert sha(cp / "state.pkl") == receipt["state_sha256"]
    with (cp / "state.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    assert finite_tree(restored) and restored["new_update"] == 1
    params, adam = P.rep(restored["params"]), P.rep(restored["adam"])
    (states, aux, keys), batch, diag = P.C.collector(8)(
        params, restored["states"], restored["aux"], restored["keys"],
        bank_states, bank_aux)
    history, _, _ = P.C.aggregate(batch, restored["history"], seed, 66)
    (next_params, next_adam), loss = P.scale_updater(64, build_optimizer())(
        params, adam, batch, batch,
        np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 66), 8)))
    jax.block_until_ready(next_params)
    deltas = P.norms(restored["params"], P.one(next_params))
    assert all(v > 0 for v in deltas.values()) and np.isfinite(np.asarray(loss)).all()
    assert finite_tree((P.one(next_params), P.one(next_adam), states, aux, keys, history))
    atomic_json(B / "FRESH_RESTORE.json", dict(
        passed=True,
        fresh_process=True,
        loaded_state_sha256=receipt["state_sha256"],
        complete_fields=sorted(restored),
        restored_adam_counters=optimizer_count(restored["adam"]),
        restored_env_rng_shapes=dict(keys=list(np.asarray(restored["keys"]).shape)),
        next_update_finite=True,
        next_actor_delta_l2=deltas["actor"],
        next_critic_delta_l2=deltas["critic"],
        next_optimizer_counters=optimizer_count(P.one(next_adam)),
        formal_does_not_inherit_smoke=True,
    ))
    notify("R2独立新进程恢复smoke通过：smoke原子SHA、完整state/Adam/RNG重载并执行下一次finite非零更新；formal从R1 u064原状态重开。")


def formal():
    items, bank_states, bank_aux = setup()
    protocol = read(B / "PROTOCOL.json")
    total = 0
    for seed in protocol["paired_seeds"]:
        if not authorize():
            atomic_json(B / "PAUSED.json", dict(passed=True, safe_boundary=True,
                                                before_next_seed=True,
                                                last_completed_seed=(protocol["paired_seeds"][protocol["paired_seeds"].index(seed)-1]
                                                                     if total else None)))
            return
        complete = train_seed(seed, items, bank_states, bank_aux)
        if not complete:
            return
        total += protocol["budget"]["new_environment_steps_per_seed"]
    atomic_json(B / "TRAINING_COMPLETE.json", dict(
        passed=True,
        runs=len(protocol["paired_seeds"]),
        seeds=protocol["paired_seeds"],
        new_environment_steps=total,
        optimizer_steps=protocol["budget"]["total_new_optimizer_steps"],
        source_r1_steps=len(protocol["paired_seeds"]) * OLD_STEPS_PER_SEED,
        no_peak_selection=True,
        old_queues_remain_paused=True,
    ))
    notify("R2三个固定seed长预算GPU训练完成；异步CPU/JS仍继续消费，不恢复旧队列。")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "formal"
    if command == "smoke":
        smoke()
    elif command == "restore-smoke":
        restore_smoke()
    elif command == "formal":
        formal()
    else:
        raise SystemExit(command)
