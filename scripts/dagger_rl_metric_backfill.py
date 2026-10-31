"""R10 passive attribution repair; immutable training sources are untouched."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.population_run_state import atomic_json, lock, sha

R = Path(__file__).resolve().parents[1]
B = R / 'runs/dagger_rl_20261024/r10_release_qualification'
OUT = B / 'measurement_v2_backfill'
PASSIVE = frozenset(('owned_self_contacts', 'other_self_contacts',
                     'owned_slide_completed', 'owned_slide_beneficial',
                     'owned_slide_harm', 'owned_net_home_distance'))


def read(path):
    return json.loads(Path(path).read_text())


def compare_natural(old, new):
    """Exact recursive equality except the six declared item attribution fields."""
    changes = []
    def visit(a, b, path=()):
        allowed = len(path) >= 3 and path[-3] == 'item_metrics' and path[-1] in PASSIVE
        if allowed:
            assert type(a) in (int, float) and type(b) in (int, float), (path, 'numeric attribution')
            if a != b: changes.append(dict(path=list(path), old=a, new=b))
            return
        assert type(a) is type(b), (path, 'type')
        if isinstance(a, dict):
            assert a.keys() == b.keys(), (path, 'keys')
            for k in a: visit(a[k], b[k], path + (k,))
        elif isinstance(a, list):
            assert len(a) == len(b), (path, 'length')
            for i, (av, bv) in enumerate(zip(a, b)): visit(av, bv, path + (i,))
        elif a != b:
            assert allowed, (path, a, b)
            changes.append(dict(path=list(path), old=a, new=b))
    visit(old, new)
    return changes


def jobs():
    final = [B / 'formal/no_update/u000'] + sorted((B / 'formal').glob('*_s*/u032'))
    middle = sorted((B / 'formal').glob('*_s*/u016'))
    assert len(final) == 7 and len(middle) == 6
    protocol = read(B / 'CPU_EVAL_PROTOCOL.json')
    return [(cp, 'dev', protocol['natural_seed_start']) for cp in final + middle] + [
        (cp, 'holdout', protocol['holdout_seed_start']) for cp in final]


def repair_job(job):
    cp, split, seed = job
    dest = OUT / cp.relative_to(B) / split
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / 'REPAIR_COMPLETE.json'
    if receipt.exists():
        result = read(receipt)
        assert sha(R / result['adopted_path']) == result['corrected_sha256']
        return result
    original = cp if split == 'dev' else cp / 'holdout'
    source = original / 'retention'
    fixed = dest / 'replay'
    (fixed / 'retention').mkdir(parents=True, exist_ok=True)
    if not (fixed / 'retention/model').exists():
        shutil.copytree(source / 'model', fixed / 'retention/model')
    old = source / 'natural_full64.json'
    assert old.exists()
    if not (fixed / 'retention/natural_full64.json').exists():
        env = dict(os.environ, NODE_OPTIONS='--require=./scripts/dagger_rl_metrics_v2_preload.js')
        with (dest / 'replay.log').open('a') as stream:
            subprocess.run([sys.executable, '-m', 'scripts.dagger_rl_metric_backfill',
                            'natural', str(fixed.relative_to(R)), str(seed)],
                           cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    new = fixed / 'retention/natural_full64.json'
    old_data, new_data = read(old), read(new)
    changes = compare_natural(old_data, new_data)
    # Full action/state/reward/event first-episode trace and Review identity exact.
    traces = []
    for pattern in ['fixed-*.trace.json', 'REVIEW_FIXED_*.json']:
        for file in sorted(source.glob(pattern)):
            other = new.parent / file.name
            assert other.exists() and sha(file) == sha(other), ('trace parity', file.name)
            traces.append(dict(name=file.name, sha256=sha(file)))
    assert traces, 'fixed trace evidence missing'
    # Per-shard verification catches errors hidden by aggregate summaries.
    for shard in sorted((source / 'natural_shards').glob('*/RESULT.json')):
        compare_natural(read(shard), read(new.parent / shard.relative_to(source)))
    archive = dest / 'legacy_v1'
    archive.mkdir(exist_ok=True)
    for name in ['natural_full64.json', 'COMPLETE.json']:
        f = source / name
        if f.exists() and not (archive / name).exists(): shutil.copy2(f, archive / name)
    # Preserve the original shards; consumers receive the internally consistent v2 set.
    if not (archive / 'natural_shards').exists():
        shutil.copytree(source / 'natural_shards', archive / 'natural_shards')
    old_hash, new_hash = sha(old), sha(new)
    for shard in sorted((new.parent / 'natural_shards').glob('*/RESULT.json')):
        atomic_json(source / shard.relative_to(new.parent), read(shard))
    atomic_json(old, new_data)
    if split == 'holdout':
        target = cp / 'retention/natural_holdout64.json'
        if not (archive / 'natural_holdout64.json').exists(): shutil.copy2(target, archive / target.name)
        compare_natural(read(target), new_data)
        atomic_json(target, new_data)
    elif (source / 'COMPLETE.json').exists():
        complete = read(source / 'COMPLETE.json')
        compare_natural(complete['natural'], new_data)
        complete['natural'] = new_data
        atomic_json(source / 'COMPLETE.json', complete)
    ticks = new_data['results']['natural_full2400']['ticks']
    result = dict(passed=True, checkpoint=str(cp.relative_to(B)), split=split,
                  source_seed_start=seed, legacy_sha256=old_hash, corrected_sha256=new_hash,
                  adopted_path=str(old.relative_to(R)), changed_passive_fields=changes,
                  exact_fixed_traces=traces, episode_invariants_exact=True,
                  executed_replay_physics_steps=ticks, reexecuted_source_episodes=64,
                  new_independent_episodes=0, optimizer_steps=0, unix=time.time())
    atomic_json(receipt, result)
    print(f"repaired {cp.parent.name}/{cp.name} {split}: {len(changes)} passive fields; {ticks} replay steps", flush=True)
    return result


def owner():
    OUT.mkdir(parents=True, exist_ok=True)
    unique = lock(OUT / 'owner.lock')
    atomic_json(OUT / 'OWNER_STATUS.json', dict(stage='WAITING_ORIGINAL_CPU_OWNER', controls_gpu=False))
    with (R / 'runs/je10_bun_20261018/evaluation.lock').open('a') as global_lock:
        fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX)
        assert read(B / 'CPU_EVALUATION_COMPLETE.json')['passed']
        plan = jobs()
        atomic_json(OUT / 'PROTOCOL.json', dict(
            schema='r10-passive-owned-attribution-repair/v1',
            frozen_training_protocol_sha256=sha(B / 'PROTOCOL.json'),
            node_preload='scripts/dagger_rl_metrics_v2_preload.js',
            preload_sha256=sha(R / 'scripts/dagger_rl_metrics_v2_preload.js'),
            metrics_sha256=sha(R / 'scripts/dagger_rl_metrics_v2.js'),
            changed_fields=sorted(PASSIVE), unchanged='all other fields and fixed actions/state/events/reward trace',
            cases=[dict(checkpoint=str(cp.relative_to(B)), split=split, seed=seed) for cp,split,seed in plan],
            reused_Hunter_Tactical=True, reexecuted_episodes=1280, new_independent_episodes=0,
            controls_gpu=False, training_invalid=False))
        atomic_json(OUT / 'OWNER_STATUS.json', dict(stage='REPLAYING_V2', controls_gpu=False))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(repair_job, plan))
        result = dict(passed=True, jobs=len(results), reexecuted_episodes=1280,
                      new_independent_episodes=0, optimizer_steps=0,
                      executed_replay_physics_steps=sum(x['executed_replay_physics_steps'] for x in results),
                      changed_passive_fields=sum(len(x['changed_passive_fields']) for x in results),
                      all_nonattribution_fields_and_fixed_traces_exact=True,
                      protocol_sha256=sha(OUT / 'PROTOCOL.json'), results=results, unix=time.time())
        atomic_json(B / 'MEASUREMENT_REPAIR_COMPLETE.json', result)
        hold = read(B / 'EVIDENCE_MEASUREMENT_HOLD.json')
        hold.update(hold=False, resolved=True, repair_sha256=sha(B / 'MEASUREMENT_REPAIR_COMPLETE.json'))
        atomic_json(B / 'EVIDENCE_MEASUREMENT_HOLD.json', hold)
        atomic_json(OUT / 'OWNER_STATUS.json', dict(stage='COMPLETE', controls_gpu=False))
        from scripts.dagger_rl_r10_run import notify
        notify('R10 v2被动归因补算完成：20×64既有源局，全部非归因字段及固定动作/状态/事件trace精确；不增加独立样本、不重训。')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'natural':
        assert os.environ.get('NODE_OPTIONS') == '--require=./scripts/dagger_rl_metrics_v2_preload.js'
        from scripts.dagger_rl_r10_cpu import natural
        natural(R / sys.argv[2], seed_start=int(sys.argv[3]))
    else:
        try: owner()
        except Exception as error:
            OUT.mkdir(parents=True, exist_ok=True)
            atomic_json(OUT / 'TECHNICAL_FAILURE.json', dict(reason=str(error).replace(str(R), '<WORKTREE>')))
            raise
