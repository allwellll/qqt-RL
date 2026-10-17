"""H8 v36 continuous-cell route-potential fixed-budget controller."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02'
os.environ.setdefault('JE09_NAMESPACE', NAMESPACE)
os.environ.setdefault('JAXBOMB_RULE', 'bun')
sys.path.insert(0, str(ROOT))
from scripts import je09_stage_h8_pbrs as h8
from scripts import je09_stage_h6 as h6

BASE = ROOT / NAMESPACE
ARM = 'bun392_bridge_pfsp8'
SMOKE_NAME = 'production_smoke_route_pbrs_continuous_v36_attempt02'
POINTS = [4, 8, 12, 16, 24, 32]


def configure():
    ctl = h8.configure()
    ctl.SCHEMA = 'je09-eight-device-stage-h8-route-pbrs-continuous-v36'
    ctl.ENTRY = 'scripts/je09_route_pbrs_continuous.py'
    ctl.SMOKE_NAME = SMOKE_NAME
    old = ctl.configure_worker

    def configure_worker(run):
        worker = old(run)
        worker_module = sys.modules['scripts.je09_stage_h5c_worker']
        worker_module.BASE = BASE
        worker_module.RUN = Path(run)
        return worker

    ctl.configure_worker = configure_worker
    return ctl


def prepare(ctl):
    from scripts.population_run_state import atomic_json, sha
    h6.prepare(ctl)
    declaration = json.loads((BASE / 'ROUND_DECLARATION.json').read_text())
    plan = json.loads((BASE / 'plan.json').read_text())
    plan.update(generations=32, checkpoints=[0, *POINTS], total_fresh_ticks=67108864,
        seed=declaration['seeds']['train'], state_schema=ctl.SCHEMA,
        route_pbrs=declaration['route_pbrs'])
    plan['campaign'].update(stage='H8-PBRS-CONTINUOUS', change=declaration['change'],
        stopping='fixed32; technical integrity failures only')
    plan['parent_selection'] = dict(checkpoint=declaration['parent'], generation=32,
        policy_sha256='b491758761b32872555555f08cf4e42f9cab120215501509a2bdbf905e64ce11',
        state_sha256='f2fcf9c44401b924776e7e7e990bceebe1660fb1397de30d1476416766716ec7',
        fresh_optimizer_rng_env=True)
    atomic_json(BASE / 'plan.json', plan)
    seal = json.loads((BASE / 'SOURCE_SEAL.json').read_text())
    seal['plan_sha256'] = sha(BASE / 'plan.json')
    names = ['scripts/je09_route_pbrs.py', 'scripts/je09_stage_h8_pbrs_ops.py',
        'scripts/je09_stage_h8_pbrs.py', 'scripts/je09_stage_h8_pbrs_declare.py',
        'scripts/je09_route_pbrs_continuous.py', 'scripts/je09_route_pbrs_continuous_declare.py',
        'tests/test_je09_route_pbrs.py']
    for name in names:
        seal['sources'][name] = sha(ROOT / name)
        target = BASE / 'inputs/sealed_sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    atomic_json(BASE / 'SOURCE_SEAL.json', seal)
    atomic_json(BASE / 'PREPARED.json', dict(passed=True, plan_sha256=sha(BASE / 'plan.json'),
        source_seal_sha256=sha(BASE / 'SOURCE_SEAL.json'), parent=declaration['parent']))
    atomic_json(BASE / 'FROZEN_BEFORE_SMOKE.json', dict(
        declaration_sha256=sha(BASE / 'ROUND_DECLARATION.json'), plan_sha256=sha(BASE / 'plan.json'),
        source_seal_sha256=sha(BASE / 'SOURCE_SEAL.json')))


def run(ctl):
    from scripts.population_run_state import atomic_json, sha
    ctl.check_seal()
    smoke = BASE / SMOKE_NAME
    (smoke / 'formal' / ARM).mkdir(parents=True, exist_ok=True)
    (smoke / 'inputs').mkdir(parents=True, exist_ok=True)
    for name in ['initial.pkl', 'initial_pool.pkl', 'frozen.pkl', 'regions.pkl', 'retention_parent.pkl']:
        shutil.copyfile(BASE / 'inputs' / name, smoke / 'inputs' / name)
    shutil.copyfile(BASE / 'regions.json', smoke / 'regions.json')
    plan = json.loads((BASE / 'plan.json').read_text())
    plan.update(generations=1, checkpoints=[0, 1], seed=json.loads(
        (BASE / 'ROUND_DECLARATION.json').read_text())['seeds']['technical'])
    atomic_json(smoke / 'plan.json', plan)
    proc = ctl.launch('worker', smoke, log_name='production_smoke.log')
    atomic_json(BASE / 'SMOKE_LAUNCH.json', dict(pid=proc.pid, devices=list(range(8))))
    if proc.wait() != 0:
        raise RuntimeError('v36 real 8-GPU smoke failed')
    rows = [json.loads(x) for x in (smoke / 'formal' / ARM / 'events.jsonl').read_text().splitlines()]
    event = rows[-1]
    assert event['finite'] and event['nonzero_update'] and event['fresh_ticks'] == 2097152
    assert event['optimizer_updates'] == 128 and event['dagger']['noneligible_label'] == 0
    marker = ctl.verify(smoke / 'formal' / ARM / 'gen000001')
    assert marker['replicas_exact'] and marker['devices'] == 8
    atomic_json(BASE / 'PRODUCTION_SMOKE_COMPLETE.json', dict(passed=True, devices=8,
        generation=1, fresh_ticks=2097152, finite=True, nonzero_update=True,
        replicas_exact=True, lambda_fixed=.05, source_seal_sha256=sha(BASE / 'SOURCE_SEAL.json')))
    assert ctl.launch('restore', smoke, smoke / 'formal' / ARM / 'gen000001',
                      cpu=True, log_name='fresh_cpu_restore.log').wait() == 0
    assert ctl.launch('replay', smoke, smoke / 'formal' / ARM / 'gen000000',
                      log_name='fresh_gpu_replay.log').wait() == 0
    replay = json.loads((smoke / 'state/FRESH_GPU_REPLAY.json').read_text())
    assert replay['next_update_exact']
    observer = subprocess.Popen(['taskset', '-c', '80-95', sys.executable,
        'scripts/je09_stage_h3_evaluate.py'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JE09_NAMESPACE=NAMESPACE),
        stdout=(BASE / 'async_evaluator.log').open('a'), stderr=subprocess.STDOUT, start_new_session=True)
    train = ctl.launch('worker', BASE, log_name='formal.log')
    atomic_json(BASE / 'TRAIN_LAUNCH.json', dict(pid=train.pid, supervisor_pid=os.getpid(),
        devices=list(range(8)), fixed_generations=32, fresh_ticks=67108864,
        evaluations_asynchronous=True, no_capability_early_stop=True,
        evaluator_pid=observer.pid, parent=json.loads((BASE / 'ROUND_DECLARATION.json').read_text())['parent']))
    first = BASE / 'formal' / ARM / 'STATE.json'
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline and not first.exists():
        if train.poll() is not None:
            raise RuntimeError('formal worker exited before first STATE')
        time.sleep(1)
    if not first.exists():
        raise TimeoutError('formal worker did not publish STATE')
    state = json.loads(first.read_text())
    assert state['generation'] >= 1 and state['finite'] and state['nonzero_update']
    atomic_json(BASE / 'FIRST_FORMAL_STATE.json', dict(pid=train.pid,
        generation=state['generation'], fresh_ticks=state['fresh_ticks'],
        optimizer_updates=state['optimizer_updates'], finite=state['finite'],
        nonzero_update=state['nonzero_update'], parent_checkpoint=BASE.as_posix()))
    print(json.dumps(dict(pid=train.pid, generation=state['generation'],
        fresh_ticks=state['fresh_ticks'], optimizer_updates=state['optimizer_updates'],
        finite=state['finite'], nonzero_update=state['nonzero_update'])), flush=True)
    while train.poll() is None:
        time.sleep(30)
    if train.wait() != 0:
        atomic_json(BASE / 'TECHNICAL_FAILURE.json', dict(exit_code=train.returncode))
        raise RuntimeError('v36 formal worker failed')
    exit_row = json.loads((BASE / 'formal' / ARM / 'TRAINER_EXIT.json').read_text())
    assert exit_row['generation'] == 32
    assert ctl.launch('restore', BASE, BASE / 'formal' / ARM / 'gen000032',
                      cpu=True, log_name='final_restore.log').wait() == 0
    atomic_json(BASE / 'CAMPAIGN_COMPLETE.json', dict(completed=True,
        generation=32, fresh_ticks=67108864, cpu_restore=True))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'restore', 'replay'])
    parser.add_argument('--run', default=NAMESPACE); parser.add_argument('--resume')
    args = parser.parse_args(); ctl = configure()
    if args.command == 'prepare': prepare(ctl)
    elif args.command == 'run': run(ctl)
    elif args.command == 'restore': ctl.restore(ROOT / args.run, ROOT / args.resume)
    else: ctl.run_worker(ROOT / args.run, ROOT / args.resume if args.resume else None,
                         args.command == 'replay')


if __name__ == '__main__':
    main()
