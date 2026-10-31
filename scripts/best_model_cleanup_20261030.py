"""Authorized staged duplicate cleanup, preserving policies and all outcomes."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts import checkpoint_retention as R

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'runs/bomb_threat_v1'
O = B / 'publish_best_20261030'
PLAN = O / 'CLEANUP_PLAN_20261030.json'
RECEIPT = O / 'CLEANUP_RECEIPT_20261030.json'
SOURCE = 'runs/bomb_threat_v1/formal/bt_slot05_c000032/u002320'


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def snapshot():
    args = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            a = (p / 'cmdline').read_bytes().split(b'\0')
            if b'-m' in a:
                module = a[a.index(b'-m') + 1].decode()
                if module.startswith('scripts.bomb_threat_'):
                    args.append([s.decode(errors='replace') for s in a[a.index(b'-m') + 1:] if s])
            elif b'node' in [Path(s.decode(errors='replace')).name.encode() for s in a if s]:
                args.append([s.decode(errors='replace') for s in a[1:] if s])
        except (OSError, IndexError):
            pass
    # Never serialize executable/absolute paths or machine process identifiers.
    return [[s for s in a if not s.startswith('/')] for a in args]


def required_pins():
    keep = {SOURCE: 'R1 four arms and slot05 common full-state source; single copy'}
    for slot in read(B / 'SLOTS.json', {}).values():
        commits = sorted((B / 'formal' / slot['run']).glob('u*/COMMITTED.json'))
        if commits:
            keep[str(commits[-1].parent.relative_to(ROOT))] = 'slot latest atomic state'
        proto = read(B / 'protocols' / (slot['run'] + '.json'), {})
        if not any(read(p).get('metrics', {}).get('finite') and
                   all(read(p).get('metrics', {}).get('update_l2', {}).get(k, 0) > 0 for k in ('actor', 'critic'))
                   for p in commits):
            for source in proto.get('sources', {}).values():
                keep[source['checkpoint']] = 'pending source until real atomic update'
        # Future chunk endpoints are needed by the already authorized handoff.
        for source in proto.get('sources', {}).values():
            if (ROOT / source['checkpoint'] / 'state.pkl').exists() and not commits:
                keep[source['checkpoint']] = 'in-flight restore source'
    for path in B.glob('formal/*/u*/state.pkl'):
        if time.time() - path.stat().st_mtime < 120:
            keep[str(path.parent.relative_to(ROOT))] = 'recent atomic save; conservative hold'
    return keep


def export_meta(path):
    with path.open() as f:
        prefix = f.read(16384)
    start = prefix.index('{', prefix.index('"meta"'))
    return json.JSONDecoder().raw_decode(prefix[start:])[0]


def describe(path, reason, kind, proof=None):
    st = path.lstat()
    assert path.is_file() and not path.is_symlink()
    return dict(path=str(path.relative_to(ROOT)), bytes=st.st_size, inode=st.st_ino,
                device=st.st_dev, mtime_ns=st.st_mtime_ns, reason=reason, kind=kind, proof=proof)


def completed_export_group(group, policy_cache):
    complete = read(group / 'COMPLETE.json', {})
    summary = read(group / 'SUMMARY.json', {})
    if not complete.get('passed') or not summary.get('checkpoint'):
        return None
    if sha(group / 'SUMMARY.json') != complete.get('summary_sha256'):
        return None
    cp = ROOT / summary['checkpoint']
    if not cp.is_relative_to(B / 'formal') or not (cp / 'policy.pt').exists():
        return None
    cm = read(cp / 'COMMITTED.json', {})
    h = summary.get('policy_sha256')
    if h != cm.get('policy_sha256'):
        return None
    if str(cp) not in policy_cache:
        policy_cache[str(cp)] = sha(cp / 'policy.pt')
    if policy_cache[str(cp)] != h:
        return None
    # Both complete natural action modes must have all eight complete shards.
    for mode in ('greedy', 'categorical'):
        for start in range(0, 64, 8):
            name = f'natural_{mode}_{start:03d}.json.outcome.json'
            if name not in complete.get('files', {}):
                return None
            outcome = read(group / name, {})
            if len(outcome.get('episodes', [])) != 8 or not all(e.get('done') for e in outcome['episodes']):
                return None
    model = group / 'model/model.json'
    if not model.exists() or export_meta(model).get('checkpoint_sha256') != h:
        return None
    return dict(checkpoint=summary['checkpoint'], policy_sha256=h,
                COMPLETE_sha256=sha(group / 'COMPLETE.json'), SUMMARY_sha256=sha(group / 'SUMMARY.json'))


def build():
    if PLAN.exists():
        return read(PLAN)
    O.mkdir(parents=True, exist_ok=True)
    old = read(B / 'STATE_RETENTION_PINS.json')
    keep = required_pins()
    args = snapshot()
    protected_hashes = set()
    # The real online catalog is captured by this task before inventory.
    catalog = read(O / 'ONLINE_CATALOG_BEFORE.json', {})
    for m in catalog.get('models', []):
        if m.get('source_sha256'):
            protected_hashes.add(m['source_sha256'])
    protected_hashes.update(read(O / 'SELECTION_PROTOCOL.json')['models'][l]['policy_sha256'] for l in ('candidate', 'baseline'))
    rows, skipped = [], []
    for p in B.glob('formal/*/u*/state.pkl'):
        cp = str(p.parent.relative_to(ROOT))
        if cp not in keep:
            rows.append(describe(p, 'superseded full state; revoke stale pin and delegate deletion to checkpoint_retention', 'state'))
    cache = {}
    for group in sorted((B / 'eval').iterdir()):
        if not group.is_dir() or not (group / 'model/model.json').exists():
            continue
        relative = str(group.relative_to(ROOT))
        if any(any(relative in a for a in argv) for argv in args):
            skipped.append(dict(path=relative, reason='live process references export group'))
            continue
        proof = completed_export_group(group, cache)
        if not proof or proof['policy_sha256'] in protected_hashes:
            skipped.append(dict(path=relative, reason='incomplete, identity unproven, published, or active paired model'))
            continue
        # A descriptor can be used by an asynchronous lineage watcher even when
        # no Node shard is alive. Conservatively retain every such reference.
        referenced = False
        for descriptor in (B / 'iter2').glob('**/models/*.json'):
            if descriptor.stat().st_size < 100000:
                if relative in descriptor.read_text():
                    referenced = True
                    break
        if referenced:
            skipped.append(dict(path=relative, reason='lineage descriptor references model'))
            continue
        for p in (group / 'model').iterdir():
            if p.is_file() and (p.suffix in ('.onnx', '.npz') or p.name in ('model.json', 'model.unsealed.json')):
                rows.append(describe(p, 'completed duplicate export; formal policy and committed SHA verified', 'export', proof))
    sizes = {}
    for name in ('formal', 'eval'):
        result = subprocess.run(['du', '-sb', str(B / name)], capture_output=True, text=True, check=True)
        sizes[name] = int(result.stdout.split()[0])
    d = dict(status='PLANNED_BEFORE_DELETION', authorization='.hermes/TASK_best_publish_and_cleanup_20261030_1530.md',
             files=rows, count=len(rows), bytes=sum(r['bytes'] for r in rows), required_pins=keep,
             retained_exports=skipped, directories_before_bytes=sizes,
             PINS_before=old, unix=time.time(), training_before=args,
             scope='bomb_threat formal state.pkl and fully completed eval duplicate exports; conservative skips remain documented')
    atomic_json(PLAN, d)
    return d


def open_audit(paths):
    if not paths:
        return set()
    p = subprocess.run(['lsof', '-nP', '-F', 'n', '--'] + [str(ROOT / x) for x in paths], capture_output=True, text=True)
    if p.returncode not in (0, 1) or p.stderr.strip():
        raise RuntimeError('lsof audit incomplete; deletion blocked')
    found = set()
    for s in p.stdout.splitlines():
        if s.startswith('n'):
            path = Path(s[1:])
            if path.is_relative_to(ROOT):
                found.add(str(path.relative_to(ROOT)))
    return found


def apply(stage):
    d = read(PLAN)
    target = O / f'CLEANUP_{stage.upper()}_20261030.json'
    if target.exists():
        return read(target)
    candidates = [r for r in d['files'] if (ROOT / r['path']).exists()]
    if stage == 'small':
        candidates = [r for r in candidates if r['kind'] == 'state'][:1] + [r for r in candidates if r['kind'] == 'export'][:3]
    else:
        assert read(O / 'CLEANUP_SMALL_HEALTH_20261030.json', {}).get('status') == 'PASS'
    audited_open = set()
    for start in range(0, len(candidates), 96):
        audited_open |= open_audit([r['path'] for r in candidates[start:start+96]])
    keep = required_pins()
    pins = read(B / 'STATE_RETENTION_PINS.json')
    remove = {str(Path(r['path']).parent) for r in candidates if r['kind'] == 'state' and r['path'] not in audited_open}
    # On the full pass revoke all obsolete pins, including old future promises.
    if stage == 'full':
        remove.update(p['checkpoint'] for p in pins['checkpoints'] if p['checkpoint'] not in keep)
    removed = [p for p in pins['checkpoints'] if p['checkpoint'] in remove and p['checkpoint'] not in keep]
    retained = {p['checkpoint']: p for p in pins['checkpoints'] if p not in removed}
    for checkpoint, reason in keep.items():
        retained[checkpoint] = dict(checkpoint=checkpoint, reason=reason)
    pins.setdefault('history', []).append(dict(unix=time.time(), authorization=d['authorization'], stage=stage, revoked=removed))
    pins.update(checkpoints=list(retained.values()), cleanup_authorization=d['authorization'])
    atomic_json(B / 'STATE_RETENTION_PINS.json', pins)
    deleted = []
    active_references = snapshot()
    for index, r in enumerate(candidates):
        if index % 96 == 0:
            active_references = snapshot()
        if r['kind'] != 'export' or r['path'] in audited_open:
            continue
        path = ROOT / r['path']
        st = path.lstat()
        if path.is_symlink() or (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns) != (r['inode'], r['device'], r['bytes'], r['mtime_ns']):
            continue
        # Recheck process references immediately before each deletion batch.
        group = str(path.parent.parent.relative_to(ROOT))
        if any(any(group in a for a in argv) for argv in active_references):
            continue
        path.unlink()
        deleted.append(r)
    result = dict(status='APPLIED_EXPORTS_STATE_DELEGATED', stage=stage, deleted_exports=deleted,
                  revoked_pins=removed, lsof_open_files=sorted(audited_open),
                  state_deletion_owner='scripts.checkpoint_retention existing watch; no process signaled', unix=time.time())
    atomic_json(target, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['plan', 'small', 'full'])
    args = p.parse_args()
    O.mkdir(parents=True, exist_ok=True)
    guard = lock(O / 'CLEANUP.lock')
    x = build() if args.stage == 'plan' else apply(args.stage)
    print(json.dumps({k: v for k, v in x.items() if k in ('status', 'stage', 'count', 'bytes')}))
