"""JE09 isolated real-drop bridge campaign controller."""
import argparse
import importlib
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAXBOMB_RULE", "bun")
from scripts.population_run_state import atomic_json, lock, sha
from scripts.je03_bun_stage1_20261011 import digest, read
from scripts.je03_bun831_20261011 import child_env, save_pickle
from scripts.je09_bun_migration import (ARM, NAMESPACE, SCHEMA, SHAPE, SOURCE_CHECKPOINT,
                                        SOURCE_NAMESPACE, SOURCE_EVALUATION, STAGE_B, STAGE_C, STAGE_D, continuation_plan, validate_shape)

BASE = ROOT / NAMESPACE
SOURCE = ROOT / SOURCE_NAMESPACE
ENTRY = "scripts/je09_bun_continuation.py"
POINTS = [4, 8] if STAGE_B else [8, 16]
if STAGE_D: POINTS=[1,4,5,8,9,12,13,16]
SMOKE_NAME = 'production_smoke_stage_c' if STAGE_C else 'production_smoke_stage_b' if STAGE_B else 'production_smoke_local_safe_v5'
if STAGE_D: SMOKE_NAME='production_smoke_stage_d'


def verify(cp, force=False):
    cp = Path(cp); marker = read(cp / "COMMITTED.json")
    assert marker["passed"] and marker["arm"] == ARM and marker["devices"] == 8
    assert marker["shape"] == SHAPE and marker["state_schema"] == SCHEMA
    assert sha(cp / "state.pkl") == marker["state_sha256"]
    assert marker["fresh_ticks"] == marker["generation"] * 2097152
    assert marker["adam_steps"] == [marker["generation"] * 128] * 8
    return marker


def source_audit():
    assert (SOURCE / SOURCE_EVALUATION).exists()
    cp=ROOT/SOURCE_CHECKPOINT
    marker=read(cp/"COMMITTED.json")
    assert marker["passed"] and sha(cp/"state.pkl")==marker["state_sha256"]
    assert marker["generation"]==(8 if STAGE_B else 2048)
    return cp,marker


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    own = lock(BASE / "prepare.lock")
    cp, marker = source_audit()
    old_plan = read(SOURCE / "plan.json")
    plan = continuation_plan(old_plan)
    plan["generations"] = max(POINTS)
    plan["checkpoints"] = [0] + POINTS
    plan["total_fresh_ticks"] = plan["generations"] * plan["fresh_ticks_per_generation"]
    plan["parent_selection"] = dict(checkpoint=str(cp.relative_to(ROOT)), generation=marker["generation"],
        reason="selected v7 g8 Stage B research parent" if STAGE_C else "selected v5 g8 Stage A candidate; preserve same-point raw Actor and Critic")
    if STAGE_D:plan['parent_selection']['reason']='selected Stage C g8 combined parent; same-point raw Actor/Critic and full dynamic pool'
    validate_shape(plan)
    for name in ["inputs", "formal", "state", "cache", "judge", "validation", "authority_je09_v1"]:
        (BASE / name).mkdir(parents=True, exist_ok=True)
    if (BASE / "plan.json").exists() and read(BASE / "plan.json") != plan:
        atomic_json(BASE / "plan.json", plan)
    elif not (BASE / "plan.json").exists():
        atomic_json(BASE / "plan.json", plan)
    for name in ["frozen.pkl", "regions.pkl"]:
        target = BASE / "inputs" / name
        if not target.exists(): shutil.copyfile(SOURCE / "inputs" / name, target)
    with (cp / "state.pkl").open("rb") as stream: saved = pickle.load(stream)
    params = {name: __import__("jax").tree.map(lambda x: x[0], saved["live"]["params"][name])
              for name in ["actor", "critic"]}
    if STAGE_C: save_pickle(BASE/'inputs/initial_pool.pkl',saved['live']['pool'])
    initial = BASE / "inputs" / "initial.pkl"
    if not initial.exists(): save_pickle(initial, params)
    if STAGE_C:
        shutil.copyfile(SOURCE/('inputs/retention_parent.pkl' if STAGE_D else 'inputs/initial.pkl'),BASE/'inputs/retention_parent.pkl')
    assert sha(cp / "policy.pt") == marker["policy_sha256"]
    for name in ["regions.json"]:
        target = BASE / name
        if not target.exists(): shutil.copyfile(SOURCE / name, target)
    source_hashes = {name: sha(cp / name) for name in ["state.pkl", "policy.pt", "ema_policy.pt", "COMMITTED.json"]}
    atomic_json(BASE / "PARENT_LINEAGE.json", dict(source_checkpoint=str(cp.relative_to(ROOT)),
        source_hashes=source_hashes, source_plan_sha256=sha(SOURCE / "plan.json"),
        source_evaluation_sha256=sha(SOURCE / SOURCE_EVALUATION),
        selected_generation=marker["generation"], optimizer_reset=True, rng_reset=True, env_reset=True))
    sources = {name: sha(ROOT / name) for name in [ENTRY, "scripts/je09_bun_worker.py",
        "scripts/je09_bun_ops.py", "scripts/je09_bun_scenes.py", "scripts/je09_bun_bridge.py",
        "scripts/je09_bun_migration.py", "scripts/je09_bun_evaluator.py", "scripts/je09_bun_cpu_smoke.py",
        "scripts/je09_web_bridge.js", "tests/je09_web_bridge_test.js", "tests/test_je09_bridge.py",
        "tests/test_je09_fixture_reward.py", "scripts/je09_drop_expert.py", "scripts/je09_expert_scan.py",
        "scripts/je09_web_stage_a.js", "tests/test_je09_drop_expert.py", "tests/je09_web_stage_a_test.js",
        "scripts/je09_web_stage_b.js", "tests/test_je09_stage_b.py", "tests/je09_web_stage_b_test.js", "scripts/je09_stage_b_scan.py"]}
    if STAGE_C:
        for name in ['scripts/je09_return_expert.py','scripts/je09_stage_c_scan.py','tests/test_je09_return_expert.py','tests/je09_web_stage_c_test.js','scripts/summarize_je09_stage_c.py']:
            sources[name]=sha(ROOT/name)
    if STAGE_D:
        for name in ['scripts/je09_chase_expert.py','scripts/je09_stage_d.py','scripts/je09_stage_d_scan.py','scripts/je09_web_stage_d.js','tests/test_je09_chase_expert.py']:
            sources[name]=sha(ROOT/name)
    for name in sources:
        target=BASE/"inputs/sealed_sources"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    atomic_json(BASE / "SOURCE_SEAL.json", dict(sources=sources, plan_sha256=sha(BASE / "plan.json"),
        source_plan_sha256=sha(SOURCE / "plan.json"), source_evaluation_sha256=sha(SOURCE / SOURCE_EVALUATION)))
    atomic_json(BASE / "PREPARED.json", dict(passed=True, parent_generation=marker["generation"],
        plan_sha256=sha(BASE / "plan.json"), source_seal_sha256=sha(BASE / "SOURCE_SEAL.json")))
    print("JE09 prepared", flush=True)


def check_seal():
    seal = read(BASE / "SOURCE_SEAL.json")
    assert seal["plan_sha256"] == sha(BASE / "plan.json")
    assert all(sha(ROOT / name) == value for name, value in seal["sources"].items())
    assert sha(SOURCE / "plan.json") == seal["source_plan_sha256"]
    assert sha(SOURCE / SOURCE_EVALUATION) == seal["source_evaluation_sha256"]


def configure_worker(run):
    worker = importlib.import_module("scripts.je09_bun_worker")
    worker.BASE = BASE; worker.RUN = Path(run); worker.ARM = ARM
    worker.DEVICE_COUNT = 8; worker.SHAPE = SHAPE.copy(); worker.STATE_SCHEMA = SCHEMA
    worker.POINTS = POINTS; worker.verify = verify
    def store(path, data):
        if Path(path).name == "state.pkl": data = dict(data, pool_storage="unreplicated; reconstructed across8")
        save_pickle(path, data)
    worker.save_pickle = store
    return worker


def launch(command, run, resume=None, cpu=False, log_name=None):
    env = child_env("" if cpu else "0,1,2,3,4,5,6,7", cpu)
    env["JAX_COMPILATION_CACHE_DIR"]=str(BASE/"cache"/("cpu" if cpu else "gpu"))
    cores = "96-111" if cpu else "0-79"
    args = ["taskset", "-c", cores, sys.executable, ENTRY, command, "--run", str(Path(run).relative_to(ROOT))]
    if resume: args += ["--resume", str(Path(resume).relative_to(ROOT))]
    log_path = BASE / (log_name or (command + ".log"))
    log = log_path.open("a")
    return subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def smoke():
    check_seal()
    if STAGE_D:
        assert read(BASE/'validation/CHASE_PRODUCTION_GRADIENT.json')['passed']
        assert read(BASE/'validation/ORACLE_AUDIT.json')['passed']
    else:
        assert read(BASE/"validation/EXPERT_PRODUCTION_GRADIENT.json")["passed"]
        assert read(BASE/"validation/CPU_BRIDGE_PRODUCTION.json")["passed"]
        if STAGE_C: assert read(BASE/'validation/RETURN_PRODUCTION_GRADIENT.json')['passed']
    run = BASE / SMOKE_NAME
    (run / "formal" / ARM).mkdir(parents=True, exist_ok=True)
    for name in (["initial.pkl", "frozen.pkl", "regions.pkl", "retention_parent.pkl", "initial_pool.pkl"] if STAGE_C else ["initial.pkl", "frozen.pkl", "regions.pkl"]):
        (run / "inputs").mkdir(exist_ok=True)
        shutil.copyfile(BASE / "inputs" / name, run / "inputs" / name)
    shutil.copyfile(BASE / "regions.json", run / "regions.json")
    plan = read(BASE / "plan.json"); plan.update(generations=1, checkpoints=[0, 1], smoke_generations=1)
    if STAGE_C: plan['seed']=read(BASE/'ROUND_DECLARATION.json')['seeds']['technical']
    atomic_json(run / "plan.json", plan)
    child = launch("worker", run, resume=None, log_name="smoke.log")
    assert child.wait() == 0
    final = run / "formal" / ARM / "gen000001"; marker = verify(final)
    rows = [read_line for read_line in (final.parent / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    event = __import__("json").loads(rows[0]); assert event["finite"] and event["nonzero_update"]
    atomic_json(BASE / "PRODUCTION_SMOKE_COMPLETE.json", dict(passed=True, final_generation=1,
        checkpoint_sha256=sha(final / "COMMITTED.json"), finite_nonzero=True, replicas_exact=True,
        fresh_ticks_per_second=event["fresh_ticks_per_second"], bridge_manifest="je09.real-drop-bridge/v1",
        smoke_run=str(run.relative_to(ROOT)),source_seal_sha256=sha(BASE/'SOURCE_SEAL.json')))


def restore(run, cp):
    worker = configure_worker(run); worker.restore_monitor(cp)
    atomic_json(BASE / "validation" / "FORMAL_RESTORE.json", dict(passed=True, checkpoint=str(Path(cp).relative_to(ROOT)),
        state_digest=verify(cp)["full_state_digest"], cpu_only=True))


def run_worker(run, cp, replay=False):
    check_seal(); lock(BASE / "gpu_worker.lock"); assert_unique_writer(run)
    configure_worker(run).worker(cp, replay=replay)


def assert_unique_writer(run):
    current_run = str(Path(run))
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            args = proc.joinpath("cmdline").read_bytes().decode().split("\0")
            same_run = current_run in args
            if proc.name != str(os.getpid()) and ENTRY in args and any(x in args for x in ["worker", "replay"]) and not same_run:
                raise RuntimeError("another JE09 GPU writer is active")
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass


def start():
    check_seal(); assert read(BASE / "PRODUCTION_SMOKE_COMPLETE.json")["passed"]
    assert read(BASE / 'PRODUCTION_SMOKE_COMPLETE.json')['source_seal_sha256']==sha(BASE/'SOURCE_SEAL.json')
    assert read(BASE / 'validation/FORMAL_RESTORE.json')['passed']
    smoke_cp=BASE/SMOKE_NAME/'formal'/ARM/'gen000001'
    assert read(BASE/'validation/FORMAL_RESTORE.json')['state_digest']==verify(smoke_cp)['full_state_digest']
    assert read(BASE / SMOKE_NAME / 'state/FRESH_GPU_REPLAY.json')['next_update_exact']
    assert not (BASE / 'formal' / ARM / 'gen000000/COMMITTED.json').exists(), 'fresh start would duplicate a writer/history'
    train = launch("worker", BASE, resume=None, log_name="formal.log")
    atomic_json(BASE / "TRAIN_LAUNCH.json", dict(pid=train.pid, devices=list(range(8)), generation=0))
    evaluator = subprocess.Popen(["taskset", "-c", "96-111", sys.executable,
        "scripts/je09_bun_evaluator.py", "--run", NAMESPACE], cwd=ROOT,
        env=child_env("", True), stdout=(BASE / "evaluate.log").open("a"),
        stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(BASE / "EVAL_LAUNCH.json", dict(pid=evaluator.pid, cuda_visible_devices=""))
    print(dict(training_pid=train.pid, evaluation_pid=evaluator.pid), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["prepare", "smoke", "start", "worker", "replay", "restore"])
    ap.add_argument("--run", default=NAMESPACE); ap.add_argument("--resume"); args = ap.parse_args()
    run = ROOT / args.run
    if args.command == "prepare": prepare()
    elif args.command == "smoke": smoke()
    elif args.command == "start": start()
    elif args.command == "restore": restore(run, ROOT / args.resume)
    elif args.command in ("worker", "replay"):
        run_worker(run, ROOT / args.resume if args.resume else None, args.command == "replay")


if __name__ == "__main__": main()
