"""One owner directly launches and verifies a bounded successor trainer."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.population_run_state import atomic_json,lock
from scripts.je09_stage_h5_declare import NAMESPACE,REPLICA
ARM='bun392_bridge_pfsp8'


def accepted_successor(base):
    """A launch/COMPLETE marker alone never satisfies handoff acceptance."""
    formal=base/'formal'/ARM
    required=[base/'PRODUCTION_SMOKE_COMPLETE.json',
        base/'production_smoke_stage_h5/state/RESTORE_MONITOR.json',
        base/'production_smoke_stage_h5/state/FRESH_GPU_REPLAY.json',
        formal/'gen000000/COMMITTED.json',formal/'STATE.json',base/'validation/H5_PRODUCTION_GRADIENT.json']
    if not all(p.exists() for p in required):return False
    smoke,cpu,gpu,entry,state,gradient=[json.loads(p.read_text()) for p in required]
    return (smoke['passed'] and smoke['replicas_exact'] and cpu['passed'] and cpu['cpu_only']
        and gpu['next_update_exact'] and entry['devices']==8 and state['generation']>=1
        and state['finite'] and state['nonzero_update'] and gradient['passed'])


def launch(namespace):
    base=ROOT/namespace
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        JE09_NAMESPACE=namespace,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    if not (base/'PREPARED.json').exists():
        subprocess.run([sys.executable,'scripts/je09_stage_h5.py','prepare'],cwd=ROOT,env=env,check=True)
    assert not (base/'TRAIN_LAUNCH.json').exists(),'never start a duplicate controller'
    return subprocess.Popen([sys.executable,'scripts/je09_stage_h5.py','run'],cwd=ROOT,env=env,
        stdout=(base/'controller.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)


def verify_started(process,base):
    while process.poll() is None:
        if accepted_successor(base):
            return True
        time.sleep(3)
    return accepted_successor(base)


def main():
    base=ROOT/NAMESPACE;owner=lock(base/'handoff.lock')
    atomic_json(base/'state/RELAY_OWNER.json',dict(unique_owner=True,mechanism='direct subprocess with actual successor acceptance',
        bounded_namespaces=[NAMESPACE,REPLICA],llm_thread_resume_required=False))
    for namespace in [NAMESPACE,REPLICA]:
        current=ROOT/namespace
        if namespace==REPLICA:
            complete=json.loads((base/'CAMPAIGN_COMPLETE.json').read_text())
            assert complete['actual_generation']==32 and complete['completed']
            assert not (base/'state/HARD_SAFETY_STOP.json').exists(),'hard safety requires repair, not automatic replication'
            memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            assert len(memory.splitlines())==8 and all(int(x)<1024 for x in memory.splitlines())
        process=launch(namespace)
        if not verify_started(process,current):
            atomic_json(base/'state/RELAY_FAILURE.json',dict(namespace=namespace,phase='technical_or_first_update',
                accepted=False,requires_attention=True,returncode=process.poll()))
            raise RuntimeError('successor did not reach a real finite eight-device update')
        atomic_json(base/'state'/('FORMAL_START_ACCEPTED.json' if namespace==NAMESPACE else 'HANDOFF_ACCEPTED.json'),
            dict(accepted=True,namespace=namespace,formal_generation_at_least=1,finite_nonzero=True,
                eight_devices=True,smoke_and_fresh_restore=True,notification_owner='single relay'))
        code=process.wait()
        terminal=current/'CAMPAIGN_COMPLETE.json'
        if code or not terminal.exists():
            atomic_json(base/'state/RELAY_FAILURE.json',dict(namespace=namespace,phase='training_terminal',
                returncode=code,requires_attention=True,accepted=False))
            raise RuntimeError('training terminal failure; retain evidence and do not relay')
    atomic_json(base/'state/RELAY_BUDGET_COMPLETE.json',dict(completed=True,
        bounded_runs=[NAMESPACE,REPLICA],total_formal_generations=64,
        automatic_successor_started_and_verified=True,further_training_not_predeclared=True))


if __name__=='__main__':main()
