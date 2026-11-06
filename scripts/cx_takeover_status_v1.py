"""Read-only training evidence; writes only CX takeover control-plane receipts."""
import datetime
import fcntl
import json
import statistics
import subprocess
import time
from pathlib import Path

from scripts.population_run_state import atomic_json

ROOT = Path('runs/cc_train_20261105')
BASE = ROOT / 'training'
OUT = ROOT / 'cx_takeover_20261106'
ARMS = {4: 'L2_FTN', 5: 'L2_ATK', 6: 'L2_S05h', 7: 'L2_SP'}


def read(path):
    return json.loads(Path(path).read_text())


def held(path):
    with path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def workers():
    rows = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')[:-1]
            if 'model_training.py' not in args:
                continue
            config = read(args[-1])
            if not config['module'].startswith('scripts.cc_league2_train'):
                continue
            protocol, arm, seed = config['args'][:3]
            rows[arm] = dict(pid=int(proc.name), config=args[-1], protocol=protocol,
                             seed=int(seed), args=args, cwd=str((proc / 'cwd').resolve()))
        except (OSError, ValueError, KeyError, UnicodeError):
            continue
    return rows


def snapshot():
    live = workers()
    value = dict(owner='CX', updated=datetime.datetime.now().astimezone().isoformat(),
                 allowed_gpus=[4, 5, 6, 7], disabled_gpus=[0, 1, 2, 3],
                 invoked_CC=False, training_restarted=False, arms={},
                 gpu_status=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                       '--format=csv,noheader'], text=True).splitlines())
    for gpu, arm in ARMS.items():
        d = BASE / 'formal' / arm
        # All completed rows are small enough to read, including the fresh-tick audit.
        text = (d / 'updates.jsonl').read_text()
        complete = text.rsplit('\n', 1)[0] if not text.endswith('\n') else text
        rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
        unique = {r['new_update']: r for r in rows}
        last = rows[-1]
        w = live.get(arm)
        protocol = read(w['protocol'] if w else read(d / 'SUPERVISOR.json')['protocol'])
        spec = dict(protocol['common'], **protocol['arms'][arm]) if protocol else {}
        cps = sorted(d.glob('u*/COMMITTED.json'))
        cp = read(cps[-1])
        source = read(Path(protocol['continuation_source']['checkpoint']) / 'COMMITTED.json') if protocol else {}
        ticks = sum(r['actual_environment_steps_this_update'] for r in unique.values())
        adam = sum(r['actual_optimizer_steps_this_update'] for r in unique.values())
        exposure = last['exposure']['bun']
        ratio = exposure['r3_candidate'][6] / exposure['r3_pressure'] if exposure['r3_pressure'] else None
        row = dict(gpu=gpu, worker=w, runtime='RUNNING' if w else 'NO_LIVE_WORKER',
                   update=last['new_update'], finite=last['finite'],
                   actor_critic_delta=last['update_l2'], environment_steps=last['environment_steps'],
                   optimizer_steps=last['optimizer_steps'], spec={k: spec.get(k) for k in
                   ('envs_per_device', 'global_minibatch', 'rollout_ticks', 'epochs', 'sil_coef')},
                   actual_ticks_per_update=last['actual_environment_steps_this_update'],
                   actual_adam_steps_per_update=last['actual_optimizer_steps_this_update'],
                   median_last20_envsteps_s=statistics.median(r['env_per_second'] for r in rows[-20:]),
                   unique_completed_updates=len(unique), duplicate_log_rows=len(rows)-len(unique),
                   counted_fresh_ticks=ticks, counted_adam_steps=adam,
                   source_environment_steps=source.get('environment_steps'),
                   source_optimizer_steps=source.get('optimizer_steps'),
                   cumulative_tick_delta=last['environment_steps']-source.get('environment_steps', last['environment_steps']),
                   cumulative_adam_delta=last['optimizer_steps']-source.get('optimizer_steps', last['optimizer_steps']),
                   committed=dict(file=str(cps[-1]), update=cp['update'], state_sha256=cp['state_sha256'],
                                  policy_sha256=cp['policy_sha256'],
                                  files_present=all((cps[-1].parent / f).is_file() for f in ('state.pkl', 'policy.pt'))),
                   gpu_lock_held=held(BASE / f'GPU{gpu}.lock'), worker_lock_held=held(d / 'worker.lock'),
                   reward_observation=dict(pressure_events=exposure['r3_pressure'],
                       pressure_candidate=exposure['r3_candidate'][6], pressure_candidate_per_event=ratio,
                       chain_candidate=exposure['r3_candidate'][5],
                       fountain_streak_tick_sum=exposure.get('r3_fountain_streak'),
                       chain_near_events=exposure.get('r3_chain_near')),
                   supervisor_record=read(d / 'SUPERVISOR.json'))
        receipt = d / f'REWARD_RECEIPT_{w["pid"]}.json' if w else None
        if receipt and receipt.exists():
            row['reward_receipt'] = read(receipt)
        if protocol:
            start = int(protocol['chunk_start_update'])
            budget = protocol['arms'][arm].get('budget', {})
            minimum = budget.get('min_updates')
            row['budget'] = dict(start_update=start, target_update=start+int(protocol['chunk_updates']),
                minimum_updates=minimum, minimum_fresh_ticks=minimum*spec['envs_per_device']*spec['rollout_ticks'] if minimum else None,
                completed_since_source=last['new_update']-int(protocol['continuation_source'].get('update', start)),
                chunk_updates=protocol['chunk_updates'], successor_authorization=protocol.get('cc', {}).get('successor'))
            row['fresh_tick_reconciliation_exact'] = row['cumulative_tick_delta'] == ticks
            row['adam_reconciliation_exact'] = row['cumulative_adam_delta'] == adam
        value['arms'][arm] = row
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'LATEST.json', value)
    atomic_json(Path('.hermes/CX_TRAIN_PROGRESS.json'), value)
    with (OUT / 'snapshots.jsonl').open('a') as log:
        log.write(json.dumps(value) + '\n')
    # Maintain the dated project report as evaluations finish, without changing training.
    report = Path('reports/cx_takeover_training_20261106.md')
    if report.exists():
        lines = ['## 持续核验（自动更新）', '', f"时间：{value['updated']}。", '',
                 '| 臂 | 实际阶段 | 更新 | envs/minibatch | 计入fresh ticks | 最近强评估 |',
                 '|---|---|---|---|---|---|']
        details = []
        for arm, row in value['arms'].items():
            points = sorted((ROOT/'eval_strong_v1/points').glob(f'{arm}_u*__seats01.json'))
            point = read(points[-1]) if points else None
            label = point['learner']['label'] + ' ' + str(point['selection_key']) if point else '待完成'
            lines.append(f"| {arm} | {row['runtime']} | {row['update']} | {row['spec']['envs_per_device']}/{row['spec']['global_minibatch']} | {row['counted_fresh_ticks']} | {label} |")
            if point:
                for mode, result in point['G_frozen'].items():
                    details.append(f"{point['learner']['label']} {mode}: {result['wins']}胜/{result['losses']}负/{result['draws']}平，交付{result['deliveries']}，因果杀{result['causal_kills']}，自死{result['self_deaths']}，携包失败{result['carry_failures']}。\n")
        lines += [''] + details + ['', '以上为固定局筛查，不替代同checkpoint新seed复测，不按泡数晋升。']
        start, end = '<!-- CX_LIVE_START -->', '<!-- CX_LIVE_END -->'
        document = report.read_text()
        if start in document and end in document:
            before, _, rest = document.partition(start)
            _, _, after = rest.partition(end)
            document = before + start + '\n' + '\n'.join(lines) + '\n' + end + after
        else:
            document += '\n' + start + '\n' + '\n'.join(lines) + '\n' + end + '\n'
        tmp = report.with_suffix('.tmp'); tmp.write_text(document); tmp.replace(report)
    return value


def main():
    import sys
    while True:
        value = snapshot()
        print(json.dumps({a: {k: r[k] for k in ('update', 'finite', 'spec', 'counted_fresh_ticks',
                          'fresh_tick_reconciliation_exact', 'committed')} for a, r in value['arms'].items()}), flush=True)
        if '--watch' not in sys.argv:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
