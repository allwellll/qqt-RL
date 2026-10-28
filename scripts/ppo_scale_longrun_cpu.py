"""Single CPU/production-JS owner, consuming only declared atomic points."""
from pathlib import Path

# Keep functions in this module's namespace: continuation evaluation rebinds B/R.
source = Path(__file__).with_name("ppo_scale_eval_functions.py").read_text()
exec(compile(source, "<PPO-scale-frozen-evaluation-functions>", "exec"), globals())

# Measurement v3 is passive; protocol retains the superseded v2 receipt.
os.environ['NODE_OPTIONS'] = '--require=./scripts/ppo_scale_metrics_preload.js'


def owner():
    unique = lock(B / "cpu_evaluator.lock")
    protocol = read(B / "PROTOCOL.json")
    supplement = None
    supplement_cp = None
    try:
        while not (B / "CPU_STOP.json").exists():
            points = [B / "formal/no_update/u000"]
            finals = []
            for seed in protocol["paired_seeds"]:
                for arm in protocol["arms"]:
                    base = B / "formal" / f"{arm}_s{seed}"
                    for u in protocol["checkpoints_by_arm"][arm]:
                        if u:
                            points.append(base / f"u{u:03d}")
                    finals.append(base / f"u{protocol['updates_by_arm'][arm]:03d}")
            pending = [cp for cp in points if (cp / "COMMITTED.json").exists()
                       and not (cp / "retention/COMPLETE.json").exists()]
            outcomes = dispatch_pending(pending)
            if any(code for _, code in outcomes):
                atomic_json(B / "CPU_TECHNICAL_FAILURE.json", dict(
                    checkpoints=[str(cp.relative_to(R)) for cp, code in outcomes if code],
                    controls_gpu=False))
                return
            supplement_points = [B / 'formal/no_update/u000',
                B / 'formal' / f'ppo_base_batch_s{protocol["paired_seeds"][0]}' / 'u008',
                B / 'formal' / f'ppo_scaled_batch_s{protocol["paired_seeds"][0]}' / 'u004', *finals]
            if supplement is not None and supplement.poll() is not None:
                if supplement.returncode:
                    atomic_json(B / 'CPU_SUPPLEMENT_FAILURE.json', dict(
                        checkpoint=str(supplement_cp.relative_to(R)), controls_gpu=False))
                    raise RuntimeError('fixed natural action-chain supplement failed')
                supplement = None
            if supplement is None:
                ready = [cp for cp in supplement_points
                         if (cp / 'retention/COMPLETE.json').exists()
                         and not (cp / 'natural_action_audit_v4/COMPLETE.json').exists()]
                if ready:
                    supplement_cp = ready[0]
                    supplement = subprocess.Popen([sys.executable, '-m',
                        'scripts.ppo_scale_natural_audit', str(supplement_cp.relative_to(R))],
                        stdout=(B / 'natural_action_supplement.log').open('a'),
                        stderr=subprocess.STDOUT)
            if (B / "TRAINING_COMPLETE.json").exists() and all(
                    (cp / "retention/COMPLETE.json").exists() for cp in finals):
                for cp in [B / "formal/no_update/u000", *finals]:
                    target = cp / "retention/natural_holdout64.json"
                    if target.exists():
                        continue
                    folder = cp / "holdout"
                    (folder / "retention").mkdir(parents=True, exist_ok=True)
                    if not (folder / "retention/model").exists():
                        shutil.copytree(cp / "retention/model", folder / "retention/model")
                    natural(folder, seed_start=read(B / "CPU_EVAL_PROTOCOL.json")["holdout_seed_start"])
                    shutil.copyfile(folder / "retention/natural_full64.json", target)
                atomic_json(B / "PRIMARY_CPU_EVALUATION_COMPLETE.json", dict(
                    passed=True, final_models=7, controls_gpu=False,
                    declared_checkpoints_complete=True, holdout_not_used_for_tuning=True))
                if not all((cp / 'natural_action_audit_v4/COMPLETE.json').exists()
                           for cp in supplement_points):
                    time.sleep(10)
                    continue
                atomic_json(B / "CPU_EVALUATION_COMPLETE.json", dict(
                    passed=True, final_models=7, controls_gpu=False,
                    declared_checkpoints_complete=True, holdout_not_used_for_tuning=True,
                    fixed_v4_categorical_greedy_supplement_models=len(supplement_points)))
                notify("PPO scale R1全部声明点CPU/JS和最终独立自然holdout完成。")
                return
            time.sleep(10)
    except Exception as error:
        atomic_json(B / "CPU_TECHNICAL_FAILURE.json", dict(
            reason=str(error).replace(str(R), "<WORKTREE>"), controls_gpu=False))
        raise
    finally:
        # Finish any already-submitted bounded CPU audit before releasing ownership.
        if supplement is not None:
            supplement.wait()
        atomic_json(B / "CPU_OWNER_EXIT.json", dict(unix=time.time(), controls_gpu=False))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        owner()
    elif sys.argv[1] == "evaluate":
        evaluate(R / sys.argv[2])
    elif sys.argv[1] == "natural":
        natural(R / sys.argv[2])
    elif sys.argv[1] == "tactical":
        from scripts import je08_bun_authority as authority
        cp = R / sys.argv[2]
        authority.OUT = B / "cpu_eval_inputs"
        authority.tactical(cp / "policy.pt", cp / "retention/tactical_v2.json")
