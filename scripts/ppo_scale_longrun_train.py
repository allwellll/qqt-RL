"""PPO-only long-run with an equal-environment batch-scale comparison.

The collector remains the frozen categorical student collector.  Teacher labels
are recorded for opportunity telemetry only; no DAgger term or offline action
data enters the PPO objective.
"""
import json
import os
import pickle
import subprocess
import sys
import time

import jax
import jax.numpy as j
import numpy as np
import optax

from scripts import dagger_rl_r4_train as C
from scripts import combat_item_engine as T
from scripts.carrier_mechanism_analysis import bank, merged, shard
from scripts.carrier_mechanism_stats import validate_bank_record
from scripts.population_run_state import atomic_json, sha
from scripts.ppo_scale_longrun_run import B, R, authorize, notify


def bind():
    C.B, C.R, C.read, C.authorize, C.notify = B, R, _read, authorize, notify
    T.B = getattr(T, "B", None)


def _read(path):
    with open(path) as stream:
        return json.load(stream)


def setup():
    bind()
    authorize()
    T.configure()
    C.install_provenance()
    assert jax.default_backend() == "gpu" and jax.local_device_count() == 8
    items = [x for x in bank() if x["record"]["split"] == "train"]
    assert len(items) == 32
    for item in items:
        validate_bank_record(item["record"])
    source = pickle.load((B / "inputs/parent.pkl").open("rb"))
    return items, source


def initial(seed, items, envs_per_device):
    n = 8 * envs_per_device
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(seed), n))
    bs = merged(items, "states")
    ba = merged(items, "aux")
    states, aux = jax.jit(T.fresh)(keys, bs, ba)
    return shard(states, 8), shard(aux, 8), shard(keys, 8), bs, ba


def rep(tree):
    return jax.tree.map(lambda x: j.stack([j.asarray(x)] * 8), tree)


def one(tree):
    return jax.tree.map(lambda x: np.asarray(x)[0], jax.device_get(tree))


def norms(before, after):
    return {k: float(np.sqrt(sum(np.sum((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)
                                  for a, b in zip(jax.tree.leaves(before[k]), jax.tree.leaves(after[k])))))
            for k in ["actor", "critic"]}


def pure_ppo_objective(params, batch):
    obs, global_state, actions, old_logp, advantage, returns, move_mask, ability_mask = batch[:8]
    move_logits, ability_logits, _, value_logits = C.E.train.net_forward(
        params, "transformer", obs, global_state)
    move_logits = j.where(move_mask, move_logits, -1e9)
    ability_logits = j.where(ability_mask, ability_logits, -1e9)
    move_logp = jax.nn.log_softmax(move_logits)
    ability_logp = jax.nn.log_softmax(ability_logits)
    selected = (move_logp[j.arange(len(old_logp)), actions[:, 0]] +
                ability_logp[j.arange(len(old_logp)), actions[:, 1]])
    ratio = j.exp(selected - old_logp)
    clipped = j.clip(ratio, .9, 1.1)
    ppo = j.maximum(-advantage * ratio, -advantage * clipped).mean()
    value = C.E.train.hl_gauss_value_loss(value_logits, returns)
    entropy = -(j.sum(j.exp(move_logp) * move_logp, -1) +
                j.sum(j.exp(ability_logp) * ability_logp, -1)).mean()
    approx_kl = (old_logp - selected).mean()
    clip_fraction = (j.abs(ratio - 1.) > .1).mean()
    loss = ppo + .5 * value - .005 * entropy
    return loss, j.array([ppo, value, entropy, approx_kl, clip_fraction])


def scale_updater(local_minibatch, optimizer):
    """The only batch treatment: local minibatch 32 versus 64."""
    def update(p, a, b, replay, key):
        b = tuple(x.reshape((-1,) + x.shape[2:]) for x in b)
        ad = b[4]
        mean = jax.lax.pmean(ad.mean(), "device")
        sq = jax.lax.pmean((ad ** 2).mean(), "device")
        ad = (ad - mean) / j.sqrt(j.maximum(sq - mean ** 2, 0) + 1e-8)
        b = b[:4] + (ad,) + b[5:]
        order = jax.random.permutation(key, len(ad)).reshape(-1, local_minibatch)

        def body(carry, indices):
            params, adam = carry
            minibatch = tuple(x[indices] for x in b)
            (loss, metrics), grad = jax.value_and_grad(
                lambda q: pure_ppo_objective(q, minibatch),
                has_aux=True)(params)
            grad = jax.lax.pmean(grad, "device")
            changes, next_adam = optimizer.update(grad, adam, params)
            next_params = optax.apply_updates(params, changes)
            values = j.concatenate([loss[None], metrics,
                                    optax.global_norm(grad)[None],
                                    optax.global_norm(grad["actor"]["heads"]["wb"])[None]])
            return (next_params, next_adam), jax.lax.pmean(values, "device")

        return jax.lax.scan(body, (p, a), order)

    return jax.pmap(update, axis_name="device")


def commit(base, update, params, adam, states, aux, keys, seed, arm,
           metrics=None, history=None, formal=True, env_steps=0):
    cp = base / f"u{update:03d}"
    cp.mkdir(parents=True, exist_ok=True)
    payload = dict(params=one(params), adam=one(adam), states=jax.device_get(states),
                   aux=jax.device_get(aux), keys=jax.device_get(keys),
                   history=history, update=update, seed=seed, arm=arm)
    C.save(cp / "state.pkl", payload)
    C.save(cp / "policy.pt", one(params)["actor"])
    atomic_json(cp / "COMMITTED.json", dict(
        passed=True, update=update, arm=arm, seed=seed,
        state_sha256=sha(cp / "state.pkl"), policy_sha256=sha(cp / "policy.pt"),
        metrics=metrics, environment_steps=env_steps,
        optimizer_steps=0 if update == 0 else int(metrics["optimizer_steps"])))
    if formal:
        atomic_json(B / "CPU_QUEUE" / f"{arm}_s{seed}_u{update:03d}.json",
                    dict(checkpoint=str(cp.relative_to(R)), update=update,
                         environment_steps=env_steps, final=False))
    return cp


def audit():
    """Bounded natural/mixed categorical audit before formal long training."""
    authorize()
    items, source = setup()
    p = rep(source["params"])
    envs = 8
    states, aux, keys, bs, ba = initial(2026102791, items, envs)
    collect = C.collector(128)
    rows = []
    totals = dict(ticks=0, sampled_release=0, greedy_release=0,
                  release_opportunities=0, contact_labels=0, banana_inventory_ticks=0,
                  glue_inventory_ticks=0, carry_ticks=0, deaths=0,
                  natural_ticks=0, real_pickup_bank_ticks=0)
    start = time.time()
    for chunk in range(8):
        (states, aux, keys), batch, diag = collect(p, states, aux, keys, bs, ba)
        parts, flags, opportunities, rewards, values, advantages, origins = map(np.asarray, diag)
        f = flags.reshape((-1, flags.shape[-1]))
        od = opportunities.reshape((-1, opportunities.shape[-1]))
        origin = origins.reshape(-1)
        row = dict(chunk=chunk + 1, ticks=int(f.shape[0]),
                   sampled_release=int(f[:, 3].sum()),
                   greedy_release=int(f[:, 4].sum()),
                   release_probability_sum=float(f[:, 5].sum()),
                   release_opportunities=int(od[:, 2].sum()),
                   contact_labels=int(od[:, 3].sum()),
                   banana_inventory_ticks=int(f[:, 1].sum()),
                   carry_ticks=int(f[:, 0].sum()),
                   deaths=int(f[:, 8].sum()),
                   natural_ticks=int((origin < 0).sum()),
                   real_pickup_bank_ticks=int((origin >= 0).sum()),
                   finite=bool(np.isfinite(rewards).all() and np.isfinite(advantages).all()))
        rows.append(row)
        for k in totals:
            if k == "ticks": totals[k] += row["ticks"]
            elif k in row:
                totals[k] += row[k]
        atomic_json(B / "ACTION_SAMPLING_AUDIT_PROGRESS.json",
                    dict(chunks=rows, totals=totals, elapsed_seconds=time.time() - start,
                         teacher_not_applied=True, deployment_greedy_not_measured=True))
    assert totals["sampled_release"] > 0, "categorical release action was not sampled in bounded audit"
    assert totals["release_opportunities"] > 0
    assert totals["natural_ticks"] > 0 and totals["real_pickup_bank_ticks"] > 0
    payload = dict(passed=True, rows=rows, totals=totals,
                   source="real student categorical native collector; mixed natural/real-pickup reset",
                   teacher_not_applied=True, offline_actions_not_used=True,
                   deployment_greedy_separate=True, finite=True,
                   environment_steps=totals["ticks"], optimizer_steps=0,
                   audit_parent_policy_sha256=sha(B / "inputs/parent.pkl"),
                   no_capability_gate=True)
    atomic_json(B / "ACTION_SAMPLING_AUDIT.json", payload)
    notify(f"PPO scale动作采样审计完成：{totals['ticks']}真实categorical步，释放采样{totals['sampled_release']}，机会{totals['release_opportunities']}，仅作可达性证据。")


def smoke():
    authorize()
    items, source = setup()
    rows = {}
    for arm, envs, local_mb in [("ppo_base_batch", 8, 32), ("ppo_scaled_batch", 16, 64)]:
        states, aux, keys, bs, ba = initial(2026102792 + envs, items, envs)
        params, adam = rep(source["params"]), rep(source["adam"])
        collect = C.collector(8)
        (end_states, end_aux, end_keys), batch, diag = collect(params, states, aux, keys, bs, ba)
        history, _, aggregation = C.aggregate(batch, None, 2026102792 + envs, 1)
        optimizer = optax.chain(optax.clip_by_global_norm(.5),
                                optax.adam(7.5e-5, eps=1e-8))
        update = scale_updater(local_mb, optimizer)
        (next_params, next_adam), loss = update(
            params, adam, batch, batch,
            np.asarray(jax.random.split(jax.random.PRNGKey(2026102793 + envs), 8)))
        jax.block_until_ready(next_params)
        p1, a1 = one(next_params), one(next_adam)
        delta = norms(source["params"], p1)
        ability = float(optax.global_norm(jax.tree.map(
            lambda x, y: x - y, p1["actor"]["heads"]["wb"], source["params"]["actor"]["heads"]["wb"])))
        assert all(v > 0 for v in delta.values()) and ability > 0
        assert np.isfinite(np.asarray(loss)).all()
        assert all(np.isfinite(v).all() for v in jax.tree.leaves((p1, a1)))
        cp = commit(B / "smoke" / arm, 1, next_params, next_adam,
                    end_states, end_aux, end_keys, 2026102792 + envs, arm,
                    metrics=dict(optimizer_steps=int(1024 / local_mb),
                                 actor_delta_l2=delta["actor"], critic_delta_l2=delta["critic"]),
                    history=history, formal=False, env_steps=int(8 * 128 * envs))
        rows[arm] = dict(finite=True, actor_delta_l2=delta["actor"],
                         critic_delta_l2=delta["critic"], ability_delta_l2=ability,
                         loss_finite=True, fresh_state_sha256=sha(cp / "state.pkl"),
                         local_minibatch=local_mb, global_minibatch=local_mb * 8,
                         rollout_batch=8 * 128 * envs)
    atomic_json(B / "TECHNICAL_SMOKE.json", dict(
        passed=True, arms=rows, source_parent_sha256=sha(B / "inputs/parent.pkl"),
        finite=True, real_actor_critic_ability_updates=True,
        fresh_process_restore_pending=True, capability_gate=False,
        formal_environment_steps=0, formal_optimizer_steps=0))
    notify("PPO scale技术smoke通过：base/scaled两批次均finite、Actor/Critic/ability真实非零；formal预算仍为0。")


def restore(reference=False):
    """Fresh-process restore parity for both smoke branches."""
    authorize()
    items, source = setup()
    results = {}
    for arm, envs, local_mb in [("ppo_base_batch", 8, 32), ("ppo_scaled_batch", 16, 64)]:
        cp = B / "smoke" / arm / "u001"
        x = pickle.load((cp / "state.pkl").open("rb"))
        assert sha(cp / "state.pkl") == _read(cp / "COMMITTED.json")["state_sha256"]
        states = x["states"]
        aux = x["aux"]
        keys = x["keys"]
        params, adam = rep(x["params"]), rep(x["adam"])
        bs = merged([z for z in items], "states")
        ba = merged([z for z in items], "aux")
        (end_states, end_aux, end_keys), batch, diag = C.collector(8)(params, states, aux, keys, bs, ba)
        history, _, _ = C.aggregate(batch, x["history"], x["seed"], 2)
        optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5, eps=1e-8))
        (next_params, next_adam), loss = scale_updater(local_mb, optimizer)(
            params, adam, batch, batch,
            np.asarray(jax.random.split(jax.random.PRNGKey(2026102800 + envs), 8)))
        jax.block_until_ready(next_params)
        results[arm] = dict(finite=bool(np.isfinite(np.asarray(loss)).all()),
                            state_digest=sha(cp / "state.pkl"),
                            full_next_state_digest=C.digest((end_states, end_aux, end_keys)),
                            optimizer_input_digest=C.digest(batch),
                            passive_history_digest=C.digest(history),
                            loss_digest=C.digest(np.asarray(loss)),
                            next_params_digest=C.digest(one(next_params)),
                            next_adam_digest=C.digest(one(next_adam)))
        assert results[arm]["finite"]
    if reference:
        atomic_json(B / "EXPECTED_RESTORE.json", dict(passed=True, arms=results))
    else:
        assert results == _read(B / "EXPECTED_RESTORE.json")["arms"], "fresh next-step digest mismatch"
        atomic_json(B / "FRESH_RESTORE.json", dict(passed=True, arms=results,
            full_env_aux_RNG_params_Adam_batch_history_loss_exact=True))
        notify("PPO scale fresh恢复通过：两种batch完整env/aux/RNG/params/Adam/batch/history下一步摘要exact。")


def formal():
    authorize()
    items, source = setup()
    protocol = _read(B / "PROTOCOL.json")
    control = B / "formal/no_update/u000"
    if not (control / "COMMITTED.json").exists():
        control.mkdir(parents=True, exist_ok=True)
        C.save(control / "state.pkl", source)
        C.save(control / "policy.pt", source['params']['actor'])
        atomic_json(control / "COMMITTED.json", dict(passed=True, update=0,
            state_sha256=sha(control/'state.pkl'), policy_sha256=sha(control/'policy.pt')))
    optimizer_template = optax.chain(optax.clip_by_global_norm(.5),
                                     optax.adam(protocol["common"]["lr"], eps=1e-8))
    pilot_seed = protocol['paired_seeds'][0]
    queue = [(pilot_seed, 'ppo_base_batch', 8), (pilot_seed, 'ppo_scaled_batch', 4)]
    queue += [(seed, arm, protocol['updates_by_arm'][arm])
              for seed in protocol['paired_seeds'] for arm in protocol['arms']]
    for seed, arm, target_update in queue:
            envs = protocol["batch"][arm]["envs_per_device"]
            local_mb = protocol["batch"][arm]["local_minibatch"]
            updates = protocol["updates_by_arm"][arm]
            base = B / "formal" / f"{arm}_s{seed}"
            base.mkdir(parents=True, exist_ok=True)
            if (base / "COMPLETE.json").exists():
                final_receipt = _read(base / "COMPLETE.json")
                assert final_receipt['state_sha256'] == sha(base / f'u{updates:03d}/state.pkl')
                continue
            params, adam = rep(source["params"]), rep(source["adam"])
            states, aux, keys, bs, ba = initial(seed, items, envs)
            start, history = 0, None
            last = sorted(base.glob("u*/COMMITTED.json"))
            if last:
                cp = last[-1].parent
                receipt = _read(cp / "COMMITTED.json")
                assert sha(cp / "state.pkl") == receipt["state_sha256"]
                x = pickle.load((cp / "state.pkl").open("rb"))
                start = int(x["update"]); params = rep(x["params"]); adam = rep(x["adam"])
                states, aux, keys, history = x["states"], x["aux"], x["keys"], x["history"]
            else:
                commit(base, 0, params, adam, states, aux, keys, seed, arm,
                       metrics=dict(optimizer_steps=0), history=None, formal=True, env_steps=0)
                atomic_json(base / "FRESH_PARENT.json", dict(
                    passed=True, parent_input_sha256=sha(B / "inputs/parent.pkl"),
                    params_exact=True, Adam_exact=True, DAgger_coefficient=0.))
            collect = C.collector(128)
            update = scale_updater(local_mb, optimizer_template)
            began = time.time()
            for u in range(start + 1, target_update + 1):
                authorize()
                iteration_start = time.monotonic()
                before = one(params)
                (states, aux, keys), batch, diag = collect(params, states, aux, keys, bs, ba)
                jax.block_until_ready(batch)
                rollout_seconds = time.monotonic()-iteration_start
                history, _, aggregation = C.aggregate(batch, history, seed, u)
                (params, adam), loss = update(
                    params, adam, batch, batch,
                    np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), u), 8)))
                jax.block_until_ready(params)
                iteration_seconds = time.monotonic()-iteration_start
                after = one(params)
                delta = norms(before, after)
                assert delta["actor"] > 0 and delta["critic"] > 0
                assert np.isfinite(np.asarray(loss)).all()
                assert all(np.isfinite(v).all() for v in jax.tree.leaves(after))
                parts, flags, od, rewards, values, adv, origins = map(np.asarray, diag)
                f = flags.reshape((-1, flags.shape[-1])); odata = od.reshape((-1, od.shape[-1]))
                origin = origins.reshape(-1)
                optimizer_steps = int((envs * 128 * 8 // (local_mb * 8)))
                env_steps = int(u * envs * 128 * 8)
                row = dict(arm=arm, seed=seed, update=u, finite=True,
                           environment_steps=env_steps, optimizer_steps=optimizer_steps,
                           cumulative_optimizer_steps=u * optimizer_steps,
                           rollout_batch=envs * 128 * 8,
                           global_minibatch=local_mb * 8,
                           update_l2=delta, elapsed_seconds=time.time() - began,
                           steps_per_second=(envs * 128 * 8) / max(iteration_seconds, 1e-9),
                           iteration_seconds=iteration_seconds, rollout_seconds=rollout_seconds,
                           update_and_telemetry_seconds=iteration_seconds-rollout_seconds,
                           startup_compile_included=(u==start+1),
                           loss=np.asarray(loss).mean((0, 1)).tolist(),
                           actor_ability_update_l2=float(optax.global_norm(jax.tree.map(
                               lambda x, y: x - y, after["actor"]["heads"]["wb"],
                               before["actor"]["heads"]["wb"]))),
                           ability_sampled=int(f[:, 3].sum()),
                           greedy_release=int(f[:, 4].sum()),
                           release_probability_sum=float(f[:, 5].sum()),
                           item_release_opportunities=int(odata[:, 2].sum()),
                           item_contact_labels=int(odata[:, 3].sum()),
                           carry_inventory_ticks=int(f[:, 1].sum()),
                           carry_ticks=int(f[:, 0].sum()),
                           deaths=int(f[:, 8].sum()),
                           source_exposure=dict(natural_ticks=int((origin < 0).sum()),
                                                real_pickup_bank_ticks=int((origin >= 0).sum())),
                           reward_components=np.sum(parts, axis=tuple(range(parts.ndim - 1))).tolist(),
                           gae_mean=float(adv.mean()), gae_variance=float(adv.var()),
                           dagger_coefficient=0., ppo_batch_source="fresh categorical learner stream only",
                           auxiliary_advantages_discarded=True,
                           actual_opponent_exposure=dict(current=0, ring=0, fixed=env_steps,
                                                         identity="JAX rule3"))
                with (base / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n"); stream.flush(); os.fsync(stream.fileno())
                commit(base, u, params, adam, states, aux, keys, seed, arm,
                       metrics=row, history=history, formal=True, env_steps=env_steps)
                atomic_json(B / "STATUS.json", dict(stage="FORMAL", **row))
                if u == 1:
                    notify(f"PPO scale首正式更新 {arm}/s{seed} u1 finite：Actor Δ{delta['actor']:.6g} Critic Δ{delta['critic']:.6g}，batch {envs*8*128}/{local_mb*8}，GPU owner继续。")
                if u in protocol["checkpoints_by_arm"][arm]:
                    notify(f"PPO scale {arm}/s{seed} u{u} 原子checkpoint已提交，{env_steps}环境步；异步CPU/JS继续。")
            if target_update < updates:
                atomic_json(base / 'SCALE_PILOT_COMPLETE.json', dict(passed=True,
                    update=target_update, environment_steps=target_update*envs*128*8,
                    adopted_as_longrun_prefix=True, capability_gate=False))
                notify(f'PPO scale配对吞吐pilot {arm}/s{seed} u{target_update}原子完成；同环境步比较，立即接续冻结长预算。')
                continue
            final = base / f"u{updates:03d}"
            atomic_json(base / "COMPLETE.json", dict(
                passed=True, arm=arm, seed=seed, update=updates,
                state_sha256=sha(final / "state.pkl"), policy_sha256=sha(final / "policy.pt"),
                environment_steps=updates * envs * 128 * 8,
                optimizer_steps=updates * optimizer_steps))
            notify(f"PPO scale固定run终态 {arm}/s{seed} u{updates}，原子SHA {sha(final/'state.pkl')[:16]}，立即接续下一run。")
    atomic_json(B / "TRAINING_COMPLETE.json", dict(
        passed=True, runs=6, environment_steps=protocol["budget"]["total_environment_steps"],
        optimizer_steps=protocol["budget"]["total_optimizer_steps"], old_continuous_resume=False))


if __name__ == "__main__":
    authorize()
    command = sys.argv[1] if len(sys.argv) > 1 else "formal"
    if command == "audit": audit()
    elif command == "smoke": smoke()
    elif command == "restore": restore()
    elif command == "restore-reference": restore(reference=True)
    elif command == "formal": formal()
    else: raise SystemExit(command)
