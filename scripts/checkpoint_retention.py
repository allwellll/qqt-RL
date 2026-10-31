"""Authorized, bounded state.pkl retention; never removes a policy or directory."""
import argparse
import json
import os
import stat
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('runs/bomb_threat_v1', 'runs/iter_pbt_v1')
TRAIN_MODULES = ('scripts.bomb_threat_train', 'scripts.bomb_threat_long_train',
                 'scripts.bomb_threat_pbt_train', 'scripts.bomb_threat_prewarm','scripts.bomb_threat_iter2_train',
                 'scripts.bomb_threat_iter2_fallback_train','scripts.bomb_threat_iter2_diagnostics_train',
                 'scripts.bomb_threat_iter2_fallback_diagnostics_train','scripts.bomb_threat_iter2_fixed_entropy_train',
                 'scripts.bomb_threat_seat5050_train','scripts.bomb_threat_threat25_train',
                 'scripts.bomb_threat_a3s_to_p0_train','scripts.bomb_threat_pbt2350_train',
                 'scripts.bomb_threat_a3sr_train','scripts.bomb_threat_a3sr_resume_train',
                 'scripts.bomb_threat_r1_train','scripts.bomb_threat_r1_progress_train',
                 'scripts.bomb_threat_r1_preserved_train','scripts.bomb_threat_reboot_worker')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def process_snapshot(root):
    processes, opened, errors = [], set(), []
    execution_cgroup = Path('/proc/self/cgroup').read_bytes()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            # This campaign's owners/workers all run under this account.
            # Cross-account /proc tables are kernel-protected; never claim
            # visibility into them or require privileges over other projects.
            if proc.stat().st_uid != os.getuid():
                continue
            if (proc/'cgroup').read_bytes() != execution_cgroup:
                continue
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'-m' in args:
                module = args[args.index(b'-m') + 1].decode()
                if module == 'scripts.iter_pbt_train':
                    raise RuntimeError('Active iter_pbt training: cannot apply all-state deletion')
                if module in TRAIN_MODULES:
                    protocol = next((x.decode() for x in args if x.startswith(b'runs/bomb_threat_v1/') and x.endswith(b'.json')), None)
                    if protocol:
                        processes.append(dict(module=module, protocol=protocol))
                elif module in ('scripts.bomb_threat_native_eval','scripts.iter_pbt_native_eval'):
                    request=next((x.decode() for x in args if x.startswith(b'runs/') and b'/eval_requests/' in x and x.endswith(b'.json')),None)
                    if request:
                        processes.append(dict(module=module,request=request))
            for fd in (proc / 'fd').iterdir():
                try:
                    target = Path(os.readlink(fd))
                    if target.name == 'state.pkl' and target.is_relative_to(root):
                        relative = str(target.relative_to(root))
                        if any(relative.startswith(s + '/') for s in SCOPES):
                            opened.add(relative)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            errors.append('unreadable_process_or_fd_table')
    if errors:
        raise RuntimeError('Open-file audit incomplete: ' + str(len(errors)) + ' inaccessible process tables')
    return processes, opened


def inventory(root, paths=None):
    result = []
    for scope in SCOPES:
        base = root / scope
        for path in (base.rglob('state.pkl') if paths is None else (p for p in paths if p.is_relative_to(base))):
            if not path.exists():continue
            if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                raise ValueError('Refusing symbolic or out-of-scope state path')
            st = path.stat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError('Non-regular state payload')
            result.append(dict(path=str(path.relative_to(root)), bytes=st.st_size,
                               inode=st.st_ino, device=st.st_dev, mtime_ns=st.st_mtime_ns))
    return sorted(result, key=lambda row: row['path'])


def build_plan(root=ROOT, processes=None, open_paths=None, now=None, inventory_paths=None):
    root = Path(root).resolve()
    if processes is None or open_paths is None:
        processes, open_paths = process_snapshot(root)
    protected = {p: ['open_file_descriptor'] for p in open_paths}
    def keep(path, reason):
        if path.exists():
            relative = str(path.relative_to(root))
            protected.setdefault(relative, []).append(reason)
    entries = []
    b = root / SCOPES[0]
    pins = read(b / 'STATE_RETENTION_PINS.json', {})
    for item in pins.get('checkpoints', []):
        cp = Path(item['checkpoint'])
        if cp.is_absolute() or '..' in cp.parts or not str(cp).startswith(str(SCOPES[0])+'/formal/'):
            raise ValueError('Invalid persistent state retention pin')
        keep(root / cp / 'state.pkl', 'explicit_authorized_retention_pin')
    round14 = read(b / 'iter2/a3sr/round14/AUTHORIZATION.json', {})
    if round14.get('status') == 'AUTHORIZED':
        keep(root / round14['donor']['checkpoint'] / 'state.pkl', 'round14_frozen_u640_donor')
        boundary = b / 'formal' / round14['slot07_boundary_run']
        commits = sorted(boundary.glob('u*/COMMITTED.json'))
        if commits:
            keep(commits[-1].parent / 'state.pkl', 'round14_preserve_replaced_slot07_latest')
    for slot in read(b / 'SLOTS.json', {}).values():
        path = b / 'protocols' / (slot['run'] + '.json')
        entries.append(('slot', read(path, {})))
    for p in processes:
        if 'protocol' in p:
            entries.append((p['module'], read(root / p['protocol'], {})))
        else:
            req=read(root/p['request'],{})
            campaign=Path(p['request']).parents[1]
            marker=root/campaign/'eval'/Path(p['request']).stem/'native/MANIFEST.json'
            native=read(marker,{})
            if native.get('critic_evaluation') is not False:
                keep(root/req['checkpoint']/'state.pkl','legacy_native_reader_draining_before_policy_only_reload')
    active_dirs = set()
    for module, protocol in entries:
        if not protocol:
            raise ValueError('Active training protocol missing')
        for name in protocol['arms']:
            run = root / protocol['namespace'] / name
            active_dirs.add(run)
            commits = [(p, read(p)) for p in sorted(run.glob('u*/COMMITTED.json'))]
            commits = [(p, m) for p, m in commits if m.get('passed')]
            if commits:
                keep(commits[-1][0].parent / 'state.pkl', 'active_run_latest_atomic')
            real_commit = any(m.get('metrics', {}).get('finite') and
                              all(m.get('metrics', {}).get('update_l2', {}).get(k, 0) > 0
                                  for k in ('actor', 'critic')) for _, m in commits)
            # A prewarm parent may still launch its fresh restore subprocess.
            if not real_commit or module == 'scripts.bomb_threat_prewarm':
                for source in protocol['sources'].values():
                    keep(root / source['checkpoint'] / 'state.pkl', 'pending_handoff_until_first_real_atomic')
    candidates = []
    now = time.time() if now is None else now
    for row in inventory(root,inventory_paths):
        path = root / row['path']
        if not (path.parent / 'COMMITTED.json').exists() and (
                path.parent.parent in active_dirs or now - path.stat().st_mtime < 60):
            keep(path, 'atomic_save_in_progress')
        if row['path'] not in protected:
            candidates.append(row)
    return dict(status='DRY_RUN',unix=now,scopes=list(SCOPES),delete_filename='state.pkl',
                candidate_count=len(candidates),candidate_bytes=sum(x['bytes'] for x in candidates),
                candidates=candidates,protected=protected,open_state_files=sorted(open_paths),
                scanned_state_files=len(candidates)+sum((root/p).exists() for p in protected),
                process_audit_scope='all same-account processes in the current execution cgroup; other containers are not inspectable')


def apply_plan(plan, root=ROOT, refresh=None):
    root = Path(root).resolve()
    deleted, skipped = [], []
    for offset in range(0, len(plan['candidates']), 16):
        batch=plan['candidates'][offset:offset+16]
        current = refresh(root) if refresh is not None else build_plan(root,inventory_paths=[root/r['path'] for r in batch])
        batch_deleted=[]
        allowed = {r['path'] for r in current['candidates']}
        for row in plan['candidates'][offset:offset+16]:
            relative = Path(row['path']); path = root / relative
            if relative.name != 'state.pkl' or not any(path.resolve().is_relative_to((root/s).resolve()) for s in SCOPES):
                raise ValueError('Deletion target outside exact authorized filename/scope')
            if row['path'] not in allowed or not path.exists():
                skipped.append(row['path']); continue
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode) or (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns) != (
                    row['inode'], row['device'], row['bytes'], row['mtime_ns']):
                skipped.append(row['path']); continue
            path.unlink()
            deleted.append(row);batch_deleted.append(row)
        if root==ROOT:
            target=root/'runs/iter_pbt_v1/RETENTION_BATCHES.jsonl'
            with target.open('a') as stream:
                stream.write(json.dumps(dict(deleted=batch_deleted,unix=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
            atomic_json(root/'runs/iter_pbt_v1/RETENTION_APPLY_PROGRESS.json',dict(status='APPLYING',deleted_count=len(deleted),
                planned_count=plan['candidate_count'],processed_count=offset+len(batch),unix=time.time()))
    return dict(status='APPLIED',deleted_count=len(deleted),deleted_bytes=sum(r['bytes'] for r in deleted),
                deleted=deleted,skipped=skipped,unix=time.time())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--apply', action='store_true'); p.add_argument('--watch', action='store_true')
    p.add_argument('--interval', type=int, default=30); args = p.parse_args()
    base = ROOT / SCOPES[1]
    guard = lock(base / 'RETENTION.lock')
    while True:
        try:
            plan = build_plan()
            target = base / ('RETENTION_DRYRUN.json' if not (base/'RETENTION_DRYRUN.json').exists() else 'RETENTION_LATEST_DRYRUN.json')
            atomic_json(target, plan)
            if args.apply:
                receipt = apply_plan(plan)
                with (base/'RETENTION_DELETIONS.jsonl').open('a') as stream:
                    stream.write(json.dumps(receipt)+'\n'); stream.flush(); os.fsync(stream.fileno())
                final = build_plan()
                atomic_json(base/'RETENTION_STATUS.json',dict(status='RUNNING' if args.watch else 'APPLIED',
                    last_deleted_count=receipt['deleted_count'],last_deleted_bytes=receipt['deleted_bytes'],
                    protected=final['protected'],remaining_candidates=final['candidate_count'],unix=time.time()))
        except Exception as error:
            atomic_json(base/'RETENTION_STATUS.json',dict(status='BLOCKED',error=str(error),
                deletion_not_attempted_after_failed_audit=True,retry_enabled=args.watch,unix=time.time()))
            if not args.watch:
                raise
            time.sleep(max(5,args.interval))
            continue
        if not args.watch:
            return
        time.sleep(max(5,args.interval))


if __name__ == '__main__':
    main()
