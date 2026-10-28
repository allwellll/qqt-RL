"""Assist-off JE09 bridge evaluation queue, isolated from JE08 authority."""
import argparse
import importlib
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", JAXBOMB_RULE="bun")
from scripts.population_run_state import atomic_json, lock, sha
from scripts.je03_bun_stage1_20261011 import read
from scripts.je09_bun_migration import ARM, NAMESPACE, SOURCE_CHECKPOINT, STAGE_B
from scripts.je03_bun831_20261011 import child_env

BASE = ROOT / NAMESPACE
POINTS = [4, 8] if STAGE_B else [8, 16]


def evaluate_point(label, policy, committed):
    evaluator = importlib.import_module("scripts.je08_bun_evaluator")
    evaluator.BASE = BASE; evaluator.RUN = BASE; evaluator.ARM = ARM; evaluator.POINTS = POINTS
    evaluator.OUT = BASE / "authority_je09_v1"
    evaluator.authority.OUT = evaluator.OUT
    evaluator.authority.RUN = BASE
    out = evaluator.OUT / label
    out.mkdir(parents=True, exist_ok=True)
    bridge_target=out/'CONTINUOUS_BRIDGE.json'
    model=evaluator.authority.export(Path(policy),out/'model')
    if not bridge_target.exists() or 'reachable_near' not in read(bridge_target)['results']:
        subprocess.run(['node','scripts/je09_web_bridge.js',str(model),str(bridge_target),
                        str(BASE/'regions.json'),'evaluation'],cwd=ROOT,env=child_env('',True),check=True)
    stage=out/'STAGE_A.json'
    if not stage.exists():
        subprocess.run(['node','scripts/je09_web_stage_a.js',str(model),str(stage),'evaluation'],cwd=ROOT,env=child_env('',True),check=True)
    holdout=out/'STAGE_A_HOLDOUT.json'
    if not holdout.exists():
        subprocess.run(['node','scripts/je09_web_stage_a.js',str(model),str(holdout),'holdout'],cwd=ROOT,env=child_env('',True),check=True)
    if STAGE_B:
        for mode,name in [('evaluation','STAGE_B.json'),('holdout','STAGE_B_HOLDOUT.json')]:
            if not (out/name).exists():
                subprocess.run(['node','scripts/je09_web_stage_b.js',str(model),str(out/name),mode],cwd=ROOT,env=child_env('',True),check=True)
    bridge=read(bridge_target)
    assert bridge['assist'] is False and not bridge['smoke'] and bridge['model_json_sha256']==sha(model)
    proof = dict(generation=committed["generation"], committed_sha256=sha(Path(policy).parent / "COMMITTED.json"))
    evaluator.evaluate(label, Path(policy), proof)
    topology=BASE/'topology_diagnostics_v1'/label/'COMPLETE.json'
    topology.parent.mkdir(parents=True,exist_ok=True)
    if not topology.exists():
        subprocess.run(['node','scripts/je08_web_diagnostics.js',str(model),str(topology),str(BASE/'regions.json')],cwd=ROOT,env=child_env('',True),check=True)
    assert read(topology)['model_json_sha256']==sha(model)
    atomic_json(out/'JE09_COMPLETE.json',dict(passed=True,assist=False,bridge_sha256=sha(bridge_target),
        authority_sha256=sha(out/'COMPLETE_WITH_BUN_TASKS.json'),diagnostics_sha256=sha(topology),stage_a_sha256=sha(stage),holdout_sha256=sha(holdout)))
    return out / "COMPLETE_WITH_BUN_TASKS.json"


def main():
    global BASE
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=NAMESPACE); args = ap.parse_args()
    BASE=ROOT/args.run
    own = lock(BASE / "authority_je09_v1" / "owner.lock")
    out = BASE / "authority_je09_v1"; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "OWNER.json", dict(cuda_visible_devices="", phase="RUNNING", unix=time.time()))
    completed = []
    # Existing JE08 parent Hunter/Tactical shards remain authoritative. Only
    # the new continuous diagnostic needs a fresh baseline in JE09 seeds.
    parent_out=out/'baseline_parent';parent_out.mkdir(exist_ok=True)
    from scripts import je08_bun_authority as authority
    parent_model=authority.export(ROOT/SOURCE_CHECKPOINT/'policy.pt',parent_out/'model')
    if not (parent_out/'CONTINUOUS_BRIDGE.json').exists() or 'reachable_near' not in read(parent_out/'CONTINUOUS_BRIDGE.json')['results']:
        subprocess.run(['node','scripts/je09_web_bridge.js',str(parent_model),str(parent_out/'CONTINUOUS_BRIDGE.json'),
                        str(BASE/'regions.json'),'evaluation'],cwd=ROOT,env=child_env('',True),check=True)
    for mode,name in [('evaluation','STAGE_A.json'),('holdout','STAGE_A_HOLDOUT.json')]:
        if not (parent_out/name).exists():
            subprocess.run(['node','scripts/je09_web_stage_a.js',str(parent_model),str(parent_out/name),mode],cwd=ROOT,env=child_env('',True),check=True)
    if STAGE_B:
        for mode,name in [('evaluation','STAGE_B.json'),('holdout','STAGE_B_HOLDOUT.json')]:
            if not (parent_out/name).exists():
                subprocess.run(['node','scripts/je09_web_stage_b.js',str(parent_model),str(parent_out/name),mode],cwd=ROOT,env=child_env('',True),check=True)
    while True:
        for generation in POINTS:
            label = f"{ARM}_g{generation:06d}"; cp = BASE / "formal" / ARM / f"gen{generation:06d}"
            marker = cp / "COMMITTED.json"
            if not marker.exists() or label in completed: continue
            try:
                evaluate_point(label, cp / "policy.pt", read(marker)); completed.append(label)
            except BlockingIOError:
                # Another isolated prefetch may still own this checkpoint.
                # Leave it pending; never claim completion or fail the owner.
                continue
            except Exception:
                atomic_json(out / f"TECHNICAL_RETRY_{generation}_{time.time_ns()}.json", dict(generation=generation))
                raise
        atomic_json(out / "QUEUE.json", dict(completed_generations=[int(x.rsplit("g", 1)[1]) for x in completed],
            pending_generations=[g for g in POINTS if f"{ARM}_g{g:06d}" not in completed], assist_off=True,
            cuda_visible_devices="", unix=time.time()))
        if len(completed) == len(POINTS):
            atomic_json(out / "EVALUATION_COMPLETE.json", dict(passed=True, completed_generations=POINTS,
                assist_off=True, unix=time.time()))
            return
        time.sleep(10)


if __name__ == "__main__": main()
