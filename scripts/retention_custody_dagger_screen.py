"""R6 paired screen: strengthen only real short-window return DAgger rows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import retention_screen as engine
from scripts.population_run_state import atomic_json, lock, sha

BASE = ROOT / "runs/retention_20261022/r6"
PRIOR = ROOT / "runs/retention_20261021/r5"
SCRIPT = "scripts/retention_custody_dagger_screen.py"
ARM = engine.ARM
POINTS = [0, 8, 16, 24, 32]
SEEDS = [2026102210, 2026102211, 2026102212]
R5_SEEDS = [2026102107, 2026102108, 2026102109]
SOURCE = PRIOR / "A0_s2026102107"
PARENT = "runs/je10_bun_20261018/control_route_v38_r19_causal_clean/formal/bun392_bridge_pfsp8/gen000032"


def declaration():
    return engine.read(BASE / "DECLARATION.json")


def _verify_common_r5_inputs():
    common_hashes = {p.name: sha(p) for p in (SOURCE / "inputs").iterdir() if p.is_file()}
    for seed in R5_SEEDS:
        peer = PRIOR / f"A0_s{seed}"
        peer_hashes = {p.name: sha(p) for p in (peer / "inputs").iterdir() if p.is_file()}
        assert peer_hashes == common_hashes, f"R5 raw inputs differ for seed {seed}"
    return common_hashes


def _checkpoint_receipt(run, generation):
    cp = run / "formal" / ARM / f"gen{generation:06d}"
    marker = engine.read(cp / "COMMITTED.json")
    assert marker["passed"] and marker["replicas_exact"] and marker["devices"] == 8
    assert marker["generation"] == generation
    assert marker["state_schema"] == engine.read(run / "plan.json")["state_schema"]
    assert sha(cp / "state.pkl") == marker["state_sha256"]
    assert sha(cp / "policy.pt") == marker["policy_sha256"]
    return dict(
        passed=True,
        generation=generation,
        state_sha256=marker["state_sha256"],
        policy_sha256=marker["policy_sha256"],
        replicas_exact=True,
        devices=8,
        adam_steps=marker["adam_steps"],
    )


def _verify_prior():
    assert engine.read(PRIOR / "STATUS.json")["state"] == "SCREEN_COMPLETE"
    assert engine.read(PRIOR / "EVALUATION_COMPLETE.json")["passed"]
    assert engine.read(PRIOR / "DIAGNOSTIC_OWNER_COMPLETE.json")["consumed_dev"] == 30
    result = engine.read(PRIOR / "RESULTS.json")
    specs = result["runs"]
    assert len(specs) == 6
    checkpoints = []
    evaluation_counts = {"dev": 0, "holdout_parent": 0, "holdout_g32": 0, "g32_diagnostics": 0}
    for row in specs:
        run = ROOT / row["namespace"]
        terminal = engine.read(run / "FIXED_RUN_TERMINAL.json")
        assert terminal["generation"] == 32 and not terminal["atomic_stop"]
        assert engine.read(run / "GPU_RELEASE_RECEIPT.json")["released"]
        checkpoints.append(dict(namespace=row["namespace"], **_checkpoint_receipt(run, 32)))
        for generation in POINTS:
            out = run / "authority_je10" / f"g{generation:06d}"
            assert engine.read(out / "COMPLETE.json")["passed"]
            evaluation_counts["dev"] += 1
        for generation in (0, 32):
            out = run / "authority_retention_test" / f"g{generation:06d}"
            assert engine.read(out / "COMPLETE.json")["passed"]
            evaluation_counts["holdout_parent" if generation == 0 else "holdout_g32"] += 1
        diagnostic = run / "authority_je10" / "g000032"
        for name in (
            "DIAGNOSIS.json", "ROUTE_FUNNEL.json", "PHASE_COUNTERFACTUAL.json",
            "FIXED_REVIEW_PHASE_ANALYSIS_20261021.json", "DAGGER_JOINT_COVERAGE.json",
            "REVIEW_FIXED_natural_full2400.json", "REVIEW_FIXED_NATIVE.json",
        ):
            assert (diagnostic / name).is_file(), f"missing consumed diagnostic: {row['namespace']}/{name}"
        evaluation_counts["g32_diagnostics"] += 1
    assert evaluation_counts == {"dev": 30, "holdout_parent": 6, "holdout_g32": 6, "g32_diagnostics": 6}
    return result, checkpoints, evaluation_counts


def _summarize_r5_routes(arm):
    totals = dict(
        fixed_episodes=0, pickups=0, window_stock_pickups=0, stock_pickup_dead=0,
        stock_pickup_native_sugar=0, stock_pickup_free_alive=0, return_started=0,
        deliveries=0, return_oracle_opportunities=0, return_oracle_matches=0,
        first_return_actions=0, first_return_matches=0, first_deviation_episodes=0,
        progress_ticks=0, target_ticks=0,
    )
    for seed in R5_SEEDS:
        run = PRIOR / f"{arm}_s{seed}"
        funnel = engine.read(run / "authority_je10/g000032/ROUTE_FUNNEL.json")["results"]
        for task in ("native_death_stock", "native_sugar_stock"):
            item = funnel[task]
            totals["fixed_episodes"] += len(item["episodes"])
            for field, target in (
                ("pickup", "pickups"), ("window_stock_pickup", "window_stock_pickups"),
                ("stock_pickup_dead", "stock_pickup_dead"),
                ("stock_pickup_native_sugar", "stock_pickup_native_sugar"),
                ("stock_pickup_free_alive", "stock_pickup_free_alive"),
                ("return_started", "return_started"), ("delivery", "deliveries"),
                ("progress_ticks", "progress_ticks"), ("target_ticks", "target_ticks"),
            ):
                totals[target] += int(item.get(field, 0))
            returning = item.get("stage_funnel", {}).get("C_RETURN", {})
            totals["return_oracle_opportunities"] += int(returning.get("oracle_opportunities", 0))
            totals["return_oracle_matches"] += int(returning.get("oracle_matches", 0))
            totals["first_return_actions"] += int(returning.get("first_actions", 0))
            totals["first_return_matches"] += int(returning.get("first_matches", 0))
            totals["first_deviation_episodes"] += sum(
                bool(ep.get("first_deviations_by_phase", {}).get("C_RETURN"))
                for ep in item["episodes"]
            )
    totals["acquisition_scope"] = "controlled fixture with real native pickup; does not prove natural inbound path"
    totals["synthetic_carry_tasks_excluded"] = True
    totals["return_oracle_agreement"] = totals["return_oracle_matches"] / max(totals["return_oracle_opportunities"], 1)
    totals["first_return_match_rate"] = totals["first_return_matches"] / max(totals["first_return_actions"], 1)
    return totals


def _phase_summary(arm):
    runs = []
    for seed in R5_SEEDS:
        run = PRIOR / f"{arm}_s{seed}"
        cf = engine.read(run / "authority_je10/g000032/PHASE_COUNTERFACTUAL.json")
        dagger = engine.read(run / "authority_je10/g000032/DAGGER_JOINT_COVERAGE.json")
        natural = cf["networks"]["actor"]["phase_action_distributions_by_source"]["natural_complete"]
        runs.append(dict(
            seed=seed,
            phase_conditioned=bool(cf["networks"]["actor"]["conditioned"]),
            actor_counterfactuals=cf["networks"]["actor"]["counterfactuals"],
            critic_counterfactuals=cf["networks"]["critic"]["counterfactuals"],
            natural_complete_phase_counts={phase: int(row["count"]) for phase, row in natural.items()},
            joint_return_fit=dagger["fit"]["kind_3"],
            duplicate_input_groups=int(dagger["conflicts"]["duplicate_input_groups"]),
            contradictory_joint_sets=int(dagger["conflicts"]["no_common_pair_groups"]),
        ))
    return runs


def freeze_r5():
    result, checkpoints, evaluation_counts = _verify_prior()
    by_arm = {arm: [r for r in result["runs"] if r["arm"] == arm] for arm in ("A0", "A1")}
    pairs = []
    for seed in R5_SEEDS:
        a0 = next(r for r in by_arm["A0"] if r["seed"] == seed)
        a1 = next(r for r in by_arm["A1"] if r["seed"] == seed)
        pairs.append(dict(
            seed=seed,
            tail_delta_pp=100 * (a1["tail"]["natural"] - a0["tail"]["natural"]),
            auc_delta_pp=100 * (a1["auc"] - a0["auc"]),
            worst_delta_pp=100 * (a1["worst"] - a0["worst"]),
            a0_curve=a0["curve"],
            a1_curve=a1["curve"],
            a0_holdout_natural=a0["holdout"]["natural"],
            a1_holdout_natural=a1["holdout"]["natural"],
        ))
    arm_summary = {}
    for arm, rows in by_arm.items():
        arm_summary[arm] = dict(
            mean_tail_natural=sum(r["tail"]["natural"] for r in rows) / len(rows),
            mean_auc=sum(r["auc"] for r in rows) / len(rows),
            mean_worst_point=sum(r["worst"] for r in rows) / len(rows),
            mean_holdout_natural=sum(r["holdout"]["natural"] for r in rows) / len(rows),
            mean_tactical=sum(r["tail"]["tactical"] for r in rows) / len(rows),
            mean_tactical_self_death=sum(r["tail"]["tactical_self"] for r in rows) / len(rows),
            mean_hunter_hard=sum(r["tail"]["hunter_hard"] for r in rows) / len(rows),
            mean_hunter_normal=sum(r["tail"]["hunter_normal"] for r in rows) / len(rows),
            mean_hunter_easy=sum(r["tail"]["hunter_easy"] for r in rows) / len(rows),
            mean_hunter_self_hard=sum(r["tail"]["hunter_self_hard"] for r in rows) / len(rows),
            mean_hunter_self_normal=sum(r["tail"]["hunter_self_normal"] for r in rows) / len(rows),
            mean_hunter_self_easy=sum(r["tail"]["hunter_self_easy"] for r in rows) / len(rows),
        )
    tail_delta = sum(x["tail_delta_pp"] for x in pairs) / len(pairs)
    auc_delta = sum(x["auc_delta_pp"] for x in pairs) / len(pairs)
    holdout = {
        arm: [round(r["holdout"]["natural"] * 64) for r in rows]
        for arm, rows in by_arm.items()
    }
    exposure = {}
    for arm in ("A0", "A1"):
        vals = []
        for seed in R5_SEEDS:
            diag = engine.read(PRIOR / f"{arm}_s{seed}/authority_je10/g000032/DIAGNOSIS.json")
            vals.append(diag["actual_opponent_exposure"]["executed_after_overrides"]["fraction_of_valid"])
        exposure[arm] = {
            key: sum(v[key] for v in vals) / len(vals)
            for key in ("current", "ring", "fixed_league")
        }
    document = dict(
        experiment_id="RETENTION-20261021-R5",
        frozen_on="2026-10-22",
        passed=True,
        sole_factor="current-visible five-phase one-hot in Actor/Critic state token",
        fixed_budget_generations=32,
        actual_fresh_ticks=result["actual_fresh_ticks"],
        checkpoint_audit=checkpoints,
        evaluation_audit=evaluation_counts,
        paired=pairs,
        paired_tail_mean_delta_pp=tail_delta,
        paired_auc_mean_delta_pp=auc_delta,
        seed_bootstrap95=result["seed_bootstrap95"],
        arms=arm_summary,
        holdout_natural_counts_per_64=holdout,
        controlled_stock_route_funnel={arm: _summarize_r5_routes(arm) for arm in ("A0", "A1")},
        phase_diagnostics={arm: _phase_summary(arm) for arm in ("A0", "A1")},
        actual_dynamic_opponent_exposure=exposure,
        successor_decision=dict(
            selected="real-custody short-window set-valued DAgger return-row multiplier",
            rationale=[
                "A1 phase interventions changed Actor and Critic outputs in all three paired seeds; A0 interventions remained zero.",
                "Across 192 controlled native-stock route episodes per arm, return first-action oracle mismatch and incomplete delivery remain observable.",
                "Natural complete holdout direction is mixed and the three-seed dev tail and AUC both favor A0; do not promote phase conditioning or choose a holdout seed.",
                "Synthetic preloaded-carry tasks are diagnostic-only and excluded from causal return evidence.",
            ],
            limitation="Controlled stock-route fixtures begin after the combat/inbound stage; their real pickup event supports pickup-to-return analysis but not natural end-to-end acquisition claims.",
        ),
        source="runs/retention_20261021/r5/RESULTS.json and verified run/checkpoint/evaluation receipts",
    )
    atomic_json(PRIOR / "FULL_COMPARISON_20261022.json", document)
    atomic_json(PRIOR / "FREEZE_RECEIPT_20261022.json", dict(
        passed=True, checkpoint_count=6, dev_points=30, holdout_parent_points=6,
        holdout_g32_points=6, diagnostic_consumptions=6,
        comparison="FULL_COMPARISON_20261022.json",
        comparison_sha256=sha(PRIOR / "FULL_COMPARISON_20261022.json"),
        no_holdout_seed_selection=True,
    ))
    atomic_json(PRIOR / "TERMINAL_NOTICE_20261022.json", dict(
        state="R5_FROZEN_SUCCESSOR_PREPARED",
        comparison="FULL_COMPARISON_20261022.json",
        successor_experiment="RETENTION-20261022-R6-CUSTODY-RETURN-DAGGER",
        successor_namespace="runs/retention_20261022/r6",
        fixed_successor_runs=6,
        no_holdout_seed_selection=True,
    ))
    return document


def prepare():
    engine.require_training_authorized()
    if (BASE / "PREPARED.json").exists():
        return engine.read(BASE / "PREPARED.json")
    assert not BASE.exists(), "R6 namespace must be fresh"
    frozen = freeze_r5()
    parent_manifest = engine.read(PRIOR / "DECLARATION.json")
    assert parent_manifest["parent"] == PARENT
    assert len(frozen["checkpoint_audit"]) == 6
    common_hashes = _verify_common_r5_inputs()
    run_specs = []
    for seed in SEEDS:
        for arm, multiplier in (("A0", 2.0), ("A1", 4.0)):
            run_specs.append(dict(
                arm=arm,
                seed=seed,
                parent=PARENT,
                multiplier=multiplier,
                namespace=f"runs/retention_20261022/r6/{arm}_s{seed}",
            ))
    d = dict(
        experiment_id="RETENTION-20261022-R6-CUSTODY-RETURN-DAGGER",
        parent=PARENT,
        parent_policy_sha256="a2f0ca17ab0b4e84a242bed243065ab3ee9ad9bcaf80b086c1b3c320e7efaa95",
        parent_source="R5 shared raw-parent A0 inputs; no R5 learner state or phase extension",
        train_seeds=SEEDS,
        generations=32,
        points=POINTS,
        runs=run_specs,
        dev_rng=20261019,
        test_rng=20261109,
        route_dev_seeds=list(range(2026102100, 2026102132)),
        route_test_seeds=list(range(2026111000, 2026111064)),
        sole_factor="Set-valued DAgger multiplier for actual-custody C_RETURN rows at ages 0..119: A0 x2, A1 x4",
        common_configuration=dict(
            phase_conditioning=False,
            core_history_fraction=0.5,
            core_error_priority=False,
            certified_return_ppo_factor=1.0,
            reward_unchanged=True,
            ppo_gae_critic_unchanged=True,
            scene_and_opponent_mixture_unchanged=True,
            replay_capacity_additions_and_selection_unchanged=True,
            return_expert_window=120,
            free_alive_autonomous_stock_remains_legal=True,
        ),
        evaluation=dict(
            owner="scripts/retention_custody_dagger_screen.py evalowner",
            points=POINTS,
            dev_natural_games=32,
            holdout_natural_games=64,
            tactical_python_v2=64,
            hunter_production_js=dict(hard=64, normal=64, easy=64),
            route_dev_seeds=list(range(2026102100, 2026102132)),
            route_holdout_seeds=list(range(2026111000, 2026111064)),
            simulator_separation=["Python Tactical", "production JS Hunter", "JAX route/training bot"],
            holdout_consumed_after_all_dev=True,
            evaluator_controls_gpu=False,
        ),
        budget=dict(
            generations_per_run=32,
            fresh_ticks_per_generation=2097152,
            fresh_ticks_per_run=67108864,
            fixed_runs=6,
            total_fresh_ticks=402653184,
        ),
        stopping="Fixed32 every run; recovery only for technical faults; no capability or evaluation gate",
        successor_rule="At each atomic g32, verify eight GPUs idle and continue with the next predeclared paired run before waiting for evaluation or writing a report.",
        prior_r5_comparison="runs/retention_20261021/r5/FULL_COMPARISON_20261022.json",
        experiment_report="reports/retention_custody_dagger_experiment_20261022.md",
        evaluation_report="reports/retention_custody_dagger_evaluation_20261022.md",
        reward_hypotheses_installed=False,
        altered=["return DAgger relative contribution on verified real-custody short-window rows only"],
        unchanged=[
            "policy observation and phase encoding", "rewards and potential shaping", "PPO and GAE",
            "Critic objective", "scene/opponent sampling", "query budget", "FIFO capacity and row split",
            "set-valued legal/safety oracle labels", "native custody event semantics", "return-expert replay",
        ],
    )
    BASE.mkdir(parents=True)
    atomic_json(BASE / "DECLARATION.json", d)
    atomic_json(BASE / "SEMANTIC_AUDIT_HOLD.json", dict(
        status="AUDIT_COMPLETE_USER_AUTHORIZED_RESUME",
        automatic_training_allowed=True,
        audit_complete=True,
        reason="Authorized successor under .hermes/r5_authorized_resume_plan.md; no excluded namespace is resumed.",
        semantic_decision="Free-alive autonomous stock remains legal; teacher/PBRS stock qualification is unchanged.",
    ))
    for spec in run_specs:
        run = ROOT / spec["namespace"]
        (run / "inputs").mkdir(parents=True)
        for name in sorted(p.name for p in (SOURCE / "inputs").iterdir() if p.is_file()):
            target = run / "inputs" / name
            target.write_bytes((SOURCE / "inputs" / name).read_bytes())
        (run / "regions.json").write_bytes((SOURCE / "regions.json").read_bytes())
        for folder in ("state", "formal", "cache", "validation", "diagnostics"):
            (run / folder).mkdir()
        plan = engine.read(SOURCE / "plan.json")
        plan.update(
            experiment_id=f"{d['experiment_id']}-{spec['arm']}-{spec['seed']}",
            seed=spec["seed"],
            generations=32,
            checkpoints=POINTS,
            total_fresh_ticks=67108864,
            state_schema=f"retention-r6-custody-return-{spec['arm']}-{spec['seed']}",
            phase_conditioning=False,
            return_dagger_weight=dict(
                multiplier=spec["multiplier"],
                eligible_source="actual custody and age < 120 from je09_stage_h5c_dagger.labels",
                set_valued=True,
                denominator="all valid accepted legal joint labels, unchanged",
                other_phase_contributions_unchanged=True,
                query_budget_changed=False,
                replay_selection_changed=False,
                reward_changed=False,
                ppo_or_gae_changed=False,
                actor_input_changed=False,
            ),
        )
        atomic_json(run / "plan.json", plan)
        atomic_json(run / "ROUND_DECLARATION.json", dict(
            **spec, evaluation=dict(seed=d["dev_rng"], points=POINTS, route_games=32,
                                    route_seed_start=d["route_dev_seeds"][0])))
        atomic_json(run / "INPUT_AUDIT.json", dict(
            passed=True,
            parent=PARENT,
            raw_parent_inputs_sha256=common_hashes,
            copied_inputs_sha256={p.name: sha(p) for p in (run / "inputs").iterdir() if p.is_file()},
            learner_state_copied=False,
            optimizer_rng_env_fifo_core_fresh=True,
            phase_conditioning=False,
        ))
    sources = {
        str(path.relative_to(ROOT)): sha(path)
        for folder, pattern in (("scripts", "*.py"), ("scripts", "*.js"),
                                ("jax_bomb", "*.py"), ("web", "*.js"))
        for path in (ROOT / folder).glob(pattern)
    }
    atomic_json(BASE / "SOURCE_SEAL.json", dict(sources=sources))
    prepared = dict(
        passed=True,
        experiment_id=d["experiment_id"],
        fresh_namespaces=True,
        fixed_runs=6,
        fixed_generations=32,
        source_seal_sha256=sha(BASE / "SOURCE_SEAL.json"),
        r5_freeze_sha256=sha(PRIOR / "FULL_COMPARISON_20261022.json"),
        common_parent_input_sha256=common_hashes,
    )
    atomic_json(BASE / "PREPARED.json", prepared)
    engine.status("PREPARED", fixed_runs=6, generations=32, controls_gpu=False)
    persist_notification(
        "R5六个原子g32及全部评估/诊断已核验冻结；R6新namespace与三配对seed固定32代方案已准备，开始技术smoke。"
    )
    return prepared


def _environment(run, cpu=False):
    return engine.env_for(run, cpu)


def persist_notification(message):
    BASE.mkdir(parents=True, exist_ok=True)
    events = BASE / "notifications"
    events.mkdir(exist_ok=True)
    stamp = time.time_ns()
    atomic_json(events / f"event_{stamp}.json", dict(
        experiment_id="RETENTION-20261022-R6-CUSTODY-RETURN-DAGGER",
        event_id=stamp,
        message=message,
        unix=time.time(),
    ))
    print(message, flush=True)
    try:
        subprocess.run(["jdme-notify", "send", message], timeout=20,
                       check=False, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def write_continuation(state, next_run=None, active_run=None):
    completed = []
    for spec in declaration()["runs"]:
        run = ROOT / spec["namespace"]
        if (run / "FIXED_RUN_TERMINAL.json").exists():
            completed.append(spec["namespace"])
    atomic_json(BASE / "CONTINUATION.json", dict(
        experiment_id="RETENTION-20261022-R6-CUSTODY-RETURN-DAGGER",
        state=state,
        completed_runs=completed,
        active_run=active_run,
        next_run=next_run,
        resume_command="scripts/retention_custody_dagger_screen.py owner",
        resume_behavior="skip terminal runs and restore a pending run from its latest atomic checkpoint",
        fixed_generations_per_run=32,
        no_capability_gate=True,
        unix=time.time(),
    ))


def launch_worker(run, command="worker", resume=None, cpu=False):
    engine.require_training_authorized(run)
    args = ["taskset", "-c", "96-111" if cpu else "0-79", sys.executable,
            SCRIPT, command, "--run", str(run.relative_to(ROOT))]
    if resume:
        args += ["--resume", str(resume.relative_to(ROOT))]
    return subprocess.Popen(
        args, cwd=ROOT, env=_environment(run, cpu),
        stdout=(run / f"{command}.log").open("a"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def configure(run):
    from scripts import retention_core_priority_screen as shared
    worker = shared.configure(run)
    installed = __import__("scripts.retention_custody_dagger", fromlist=["install"]).install(run)
    atomic_json(run / "state/DAGGER_INSTALL.json", installed)
    return worker


def evaluation_job(run, split, generation):
    d = declaration()
    env = _environment(run, True)
    env.update(
        JE10_ROUTE_PARALLEL="1",
        JE10_DIAGNOSTIC_REPORT=d["evaluation_report"],
        JE10_EVAL_SEED=str(d["dev_rng"] if split == "dev" else d["test_rng"]),
        JE10_ROUTE_GAMES="32" if split == "dev" else "64",
        JE10_ROUTE_SEED_START=str(d["route_dev_seeds"][0] if split == "dev" else d["route_test_seeds"][0]),
        JE10_EVAL_NAMESPACE="authority_je10" if split == "dev" else "authority_retention_test",
    )
    return subprocess.run(
        [sys.executable, SCRIPT, "evalchild", "--run", str(run.relative_to(ROOT)),
         "--split", split, "--generation", str(generation)],
        cwd=ROOT, env=env,
        stdout=(run / f"{split}_eval.log").open("a"), stderr=subprocess.STDOUT,
    ).returncode


def _configure_engine():
    engine.BASE = BASE
    engine.declaration = declaration
    engine.spawn_worker = launch_worker
    engine.notify = persist_notification


def owner():
    unique = lock(BASE / "owner.lock")
    engine.require_training_authorized()
    engine.check_seal()
    d = declaration()
    first_pending = next((spec["namespace"] for spec in d["runs"]
                          if not (ROOT / spec["namespace"] / "TRAINING_COMPLETE.json").exists()), None)
    write_continuation("STARTING", next_run=first_pending)
    engine.status("TECHNICAL_SMOKE", fixed_runs=6, controls_gpu=False)
    for spec in d["runs"][:2]:
        engine.smoke(spec)
        engine.notify(
            f"R6 {spec['arm']}技术smoke通过：finite、Actor/Critic非零、原子保存/新CPU进程恢复；没有能力门禁。"
        )
    evaluator = subprocess.Popen(
        ["taskset", "-c", "80-95", sys.executable, SCRIPT, "evalowner"],
        cwd=ROOT, env=_environment(BASE, True),
        stdout=(BASE / "evaluation.log").open("a"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    atomic_json(BASE / "EVALUATOR_LAUNCH.json", dict(unique_owner=True, controls_gpu=False, unix=time.time()))
    atomic_json(BASE / "OWNER_LAUNCH.json", dict(unique_owner=True, fixed_runs=6, generations=32, unix=time.time()))
    engine.notify("R6正式筛选已接续：三配对seed、A0×2对A1×4、每run固定32代；CPU/JS评估独立跟随检查点。")
    from scripts.retention_core_priority_screen import checkpoint_receipt

    for index, spec in enumerate(d["runs"]):
        run = ROOT / spec["namespace"]
        out = run / "formal" / ARM
        if (run / "TRAINING_COMPLETE.json").exists():
            continue
        engine.wait_idle()
        engine.check_seal()
        engine.status("TRAINING", run=spec["namespace"], arm=spec["arm"], seed=spec["seed"])
        write_continuation("RUN_ACTIVE", active_run=spec["namespace"])
        for attempt in range(3):
            latest = out / "LATEST_CHECKPOINT.json"
            resume = ROOT / engine.read(latest)["checkpoint"] if latest.exists() else None
            proc = launch_worker(run, resume=resume)
            atomic_json(run / "OWNER_STATE.json", dict(
                attempt=attempt,
                restore_checkpoint=str(resume.relative_to(ROOT)) if resume else None,
                owner_active=True,
                unix=time.time(),
            ))
            announced = False
            while proc.poll() is None:
                atomic_json(BASE / "OWNER_HEARTBEAT.json", dict(
                    state="TRAINING", run=spec["namespace"], arm=spec["arm"],
                    seed=spec["seed"], attempt=attempt, unix=time.time()))
                state = out / "STATE.json"
                committed = out / "gen000001/COMMITTED.json"
                if state.exists() and committed.exists() and not announced:
                    row = engine.read(state)
                    engine.validate_update(row)
                    proof = checkpoint_receipt(run, 1)
                    install = engine.read(run / "state/DAGGER_INSTALL.json")
                    assert install["multiplier"] == spec["multiplier"]
                    atomic_json(run / "FORMAL_NONZERO_CONFIRMED.json", dict(
                        passed=True,
                        generation=row["generation"],
                        finite=row["finite"],
                        nonzero_update=row["nonzero_update"],
                        actor_update_l2=row["actor_update_l2"],
                        critic_update_l2=row["critic_update_l2"],
                        fresh_ticks=row["fresh_ticks"],
                        first_atomic_checkpoint=proof,
                        dagger_install=install,
                    ))
                    engine.notify(
                        f"R6 {spec['arm']} seed{spec['seed']}正式首代finite/Actor-Critic真实非零更新已确认，原子SHA通过；8卡训练运行，CPU评估独立。"
                    )
                    announced = True
                time.sleep(10)
            if proc.returncode == 0 and (run / "TRAINING_COMPLETE.json").exists():
                break
            atomic_json(run / f"TECHNICAL_ATTEMPT_{attempt}.json", dict(exit_code=proc.returncode))
            engine.notify(f"R6 {spec['arm']} seed{spec['seed']}技术故障，保留证据并从最近原子checkpoint恢复。")
            engine.wait_idle()
        else:
            raise RuntimeError("R6 technical retries exhausted")
        terminal = engine.read(out / "TRAINER_EXIT.json")
        assert terminal["generation"] == 32 and not terminal["atomic_stop"]
        engine.validate_update(engine.read(out / "STATE.json"))
        atomic_json(run / "FIXED_RUN_TERMINAL.json", dict(
            terminal, atomic_checkpoint=checkpoint_receipt(run, 32),
            run_budget_generations=32, successor_index=index + 1 if index + 1 < len(d["runs"]) else None,
        ))
        engine.wait_idle()
        atomic_json(run / "GPU_RELEASE_RECEIPT.json", dict(
            released=True, consecutive_idle_samples=2, resource=engine.read(BASE / "RESOURCE.json")))
        successor = d["runs"][index + 1]["namespace"] if index + 1 < len(d["runs"]) else "all fixed R6 runs complete; CPU/JS evaluation continues"
        engine.notify(
            f"R6 {spec['arm']} seed{spec['seed']}固定g32终态，原子state/policy SHA通过、8卡释放；后继：{successor}。"
        )
        write_continuation("RUN_TERMINAL", next_run=(
            d["runs"][index + 1]["namespace"] if index + 1 < len(d["runs"]) else None))
    engine.status("TRAINING_COMPLETE_EVALUATING", fixed_runs=6, generations=32)
    write_continuation("TRAINING_COMPLETE_EVALUATING")
    if evaluator.wait() != 0:
        raise RuntimeError("R6 CPU/JS evaluator failed")
    from scripts import retention_exposure_screen as results
    results.BASE = BASE
    results.declaration = declaration
    summary = results.results()
    atomic_json(BASE / "SCREEN_COMPLETE.json", summary)
    atomic_json(BASE / "OWNER_COMPLETE.json", dict(
        passed=True, fixed_runs=6, generations_per_run=32,
        evaluation_complete=True, controls_gpu=False, unix=time.time()))
    engine.status("SCREEN_COMPLETE", fixed_runs=6, generations=32)
    write_continuation("SCREEN_COMPLETE")
    engine.notify("R6 六run固定筛选与全部异步评估已完成；冻结seed配对结果，未使用能力门禁。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "owner", "worker", "restore", "evalowner", "evalchild"))
    parser.add_argument("--run")
    parser.add_argument("--resume")
    parser.add_argument("--split")
    parser.add_argument("--generation", type=int)
    args = parser.parse_args()
    _configure_engine()
    if args.command == "prepare":
        prepare()
    elif args.command == "owner":
        owner()
    elif args.command in ("evalowner", "evalchild"):
        from scripts import retention_evaluate as evaluator
        if args.command == "evalowner":
            evaluator.child = evaluation_job
            evaluator.owner()
        else:
            evaluator.evaluate_child(ROOT / args.run, args.split, args.generation)
    else:
        run = ROOT / args.run
        engine.require_training_authorized(run)
        engine.check_seal()
        worker = configure(run)
        if args.command == "restore":
            worker.restore_monitor(ROOT / args.resume)
        else:
            worker.worker(ROOT / args.resume if args.resume else None)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if BASE.exists():
            atomic_json(BASE / "TECHNICAL_FAILURE.json", dict(
                error=type(error).__name__, message=str(error), command=sys.argv[1] if len(sys.argv) > 1 else None,
                controls_training=False, unix=time.time()))
            try:
                _configure_engine()
                write_continuation("TECHNICAL_FAILURE_REQUIRES_RECOVERY")
                engine.notify(f"R6技术故障：{type(error).__name__}，已保留状态和最近原子点；不作能力判停。")
            except Exception:
                pass
        raise
