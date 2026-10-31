"""Bounded, passive production-JS recheck; never controls training or publication."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.r3_items_eval_owner import summarize
from scripts.r1_fresh_tau_0735 import paired

OUT = Path('runs/r3_ownside_recheck_1620_20261031')
OLD = Path('runs/v2_ownside_selection_20261031')
REPORT = Path('reports/r3_ownside_recheck_1620_20261031.md')


def read(p):
    return json.loads(Path(p).read_text())


def identity(checkpoint, model_json):
    cp, f = Path(checkpoint), Path(model_json)
    commit, doc = read(cp/'COMMITTED.json'), read(f)
    assert commit['passed'] and sha(cp/'policy.pt') == commit['policy_sha256']
    assert doc['meta']['checkpoint_sha256'] == commit['policy_sha256']
    graph = Path(doc['meta']['onnx_path'])
    assert sha(graph) == doc['meta']['onnx_sha256']
    return dict(checkpoint=str(cp), policy_sha256=commit['policy_sha256'],
                model_json=str(f), model_sha256=sha(f), onnx_path=str(graph),
                onnx_sha256=sha(graph), update=int(cp.name[1:]))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():
        return read(OUT/'PROTOCOL.json')
    old = read(OLD/'PROTOCOL.json')
    assert all(sha(Path(f)) == h for f, h in read(OLD/'SOURCE_SEAL.json').items())
    ms = {'baseline': identity(old['models']['r3_a_u3040']['checkpoint'],
                             read(OLD/'MODELS.json')['r3_a_u3040']['model_json'])}
    for arm, update in [('a', 3760), ('b', 3600)]:
        run = f'r3_{arm}_ownside_c000001'
        cp = Path('runs/r3_items_route_20261030/formal')/run/f'u{update:06d}'
        f = Path('runs/r3_items_route_20261030/evaluation_ownside_v1')/f'{run}_u{update:06d}_holdout/model/model.json'
        ms[f'r3_{arm}_u{update}'] = identity(cp, f)
    holdout = read('runs/r3_items_route_20261030/OWNSIDE_EVALUATION_PROTOCOL.json')
    p = dict(id='R3-OWNSIDE-RECHECK-1620-20261031', status='FROZEN',
        authorization='user 2026-10-31 16:20', models=ms,
        conditions={k: v for k, v in old['conditions'].items() if k.startswith('hunter_')},
        holdout=dict(holdout['holdout'], tier='hard', opponent='bun.hunter', games=256),
        holdout_modes=['greedy', 'categorical'], fresh_mode='greedy',
        counts=dict(fresh=2304, fresh_reused=768, holdout=512,
                    holdout_greedy_reused=128, expected_new_games=1920),
        candidate_order=['r3_a_u3760', 'r3_b_u3600'], selection=old['selection'],
        source_sha256=read(OLD/'SOURCE_SEAL.json'),
        cpu_cores=list(range(72, 84)), workers=12, cpu_only=True,
        engine='production_JS', ruleset='bun-native-v2', teacher=False, horizon=2400,
        no_publish=True, no_training_control=True, shard_size=8,
        holdout_count_amendment=dict(parent_protocol=holdout['id'],
            original='128 games/mode, 64 per seat; never combine modes as independent games',
            this_request='256 games/mode, 128 per seat; first128 is the original matched prefix',
            seed_extension='same frozen seed start, indices128..255 added; no seed replacement'),
        reuse='original accepted source files retained and hashed, no new rollout for baseline fresh or existing parent greedy holdout',
        limitations=['single map806, repeated16 own-side spawn geometries',
                     'Sim.step, not human frameStep', 'frozen fresh list already used; not a new independent holdout'], unix=time.time())
    atomic_json(OUT/'PROTOCOL.json', p)
    atomic_json(OUT/'TASK_CONSUMED.json', dict(status='FROZEN_CPU_READY', protocol_sha256=sha(OUT/'PROTOCOL.json'), unix=time.time()))
    return p


def jobs(p):
    hold = [dict(kind='holdout', label='baseline', condition='holdout', mode=m,
                 start=i, **p['holdout']) for m in p['holdout_modes'] for i in range(0, 256, 8)]
    fresh = [dict(kind='fresh', label=label, condition=c, mode='greedy', start=i, **v)
             for i in range(0, 256, 8) for c, v in p['conditions'].items() for label in p['models']]
    return hold + fresh


def target(j):
    return OUT/'shards'/j['kind']/j['condition']/j['label']/j['mode']/f"shard_{j['start']:03d}.json"


def reuse(j):
    if j['label'] == 'baseline' and j['kind'] == 'fresh':
        f = OLD/'shards'/j['condition']/'r3_a_u3040'/f"shard_{j['start']:03d}.json"
        m = read(OLD/'MODELS.json')['r3_a_u3040']
        return f, m
    if j['kind'] == 'holdout' and j['mode'] == 'greedy' and j['start'] < 128:
        root = Path('runs/r3_selfplay_ramp_20261031/evaluation/parent')
        return root/'hunter_hard'/f"shard_{j['start']:03d}.json", read(root/'model/IDENTITY.json')
    return None


def validate(j, p, f, model):
    a, b = read(f), read(str(f)+'.outcome.json')
    doc = read(model['model_json'])
    canonical = p['models'][j['label']]
    assert sha(Path(model['model_json'])) == model['model_sha256'] == a['model_json_sha256']
    assert doc['meta']['checkpoint_sha256'] == canonical['policy_sha256']
    assert doc['meta']['onnx_sha256'] == canonical['onnx_sha256']
    assert sha(Path(doc['meta']['onnx_path'])) == canonical['onnx_sha256']
    assert b['engine'] == 'production_JS' and b['rule_version'] == 'bun-native-v2'
    assert not b['teacher'] and not a.get('assist') and b['mode'] == j['mode']
    assert b['simulator_sha256'] == p['source_sha256']['web/sim-bun-native-v2.js']
    assert sha(Path(j['spawn_manifest'])) == j['spawn_sha256']
    rows = b['episodes']
    assert len(rows) == len(a['results']['natural_full2400']['episodes']) == 8
    assert [r['seed'] for r in rows] == list(range(j['seed']+j['start'], j['seed']+j['start']+8))
    from scripts.bun_spawn_side import validate as validate_side
    for i, (r, e) in enumerate(zip(rows, a['results']['natural_full2400']['episodes'])):
        assert r['seat'] == (j['start']+i)%2 and r['done'] and 0 < r['ticks'] <= 2400
        assert (r['seed'], r['seat'], r['ticks']) == (e['seed'], e['team'], e['ticks'])
        assert r['spawn_side_version'] == 'own-side-native-v1'
        validate_side(r['spawn_cells']); validate_side(r['respawn_cells'])
        assert 'r3' in r and 'antifarm' in r and 'resolved_bombs' in r['r3']
    return rows


def accepted(j, p):
    proof = Path(str(target(j))+'.ACCEPTED.json')
    if not proof.exists():
        return None
    a = read(proof)
    assert a['protocol_sha256'] == sha(OUT/'PROTOCOL.json')
    assert all(sha(Path(k)) == v for k, v in a['files'].items())
    return validate(j, p, Path(a['source_target']), a['model'])


def collect(j, p, core):
    f = target(j); f.parent.mkdir(parents=True, exist_ok=True)
    guard = lock(Path(str(f)+'.lock'))
    if accepted(j, p) is not None:
        return
    reused = reuse(j)
    if reused:
        source, model = reused
        proof = read(str(source)+'.ACCEPTED.json')
        assert all(sha(Path(k)) == v for k, v in proof['files'].items())
    else:
        source, model = f, p['models'][j['label']]
        from scripts.r3_items_eval_owner import running_shard
        while running_shard(f):
            time.sleep(3)
        if not (f.exists() and Path(str(f)+'.outcome.json').exists()):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', QQT_STOCK_RULE='stock-window-v1',
                BUN_V2_RULE='bun-native-v2', BUN_V2_TIER=j['tier'], BUN_V2_OPPONENT=j['opponent'],
                JE10_ROUTE_TASK='natural_full2400', JE10_ROUTE_GAMES=str(j['games']),
                JE10_ROUTE_START=str(j['start']), JE10_ROUTE_END=str(j['start']+8),
                JE10_ROUTE_SEED_START=str(j['seed']))
            with f.with_suffix('.log').open('ab') as log:
                subprocess.run(['taskset', '-c', str(core), 'node', 'scripts/r3_items_candidate_eval.js',
                    j['mode'], model['model_json'], str(f),
                    'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json', j['spawn_manifest']],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    validate(j, p, source, model)
    atomic_json(Path(str(f)+'.ACCEPTED.json'), dict(status='PASS', job=j, reused=bool(reused),
        source_target=str(source), model=model, protocol_sha256=sha(OUT/'PROTOCOL.json'),
        files={str(z): sha(z) for z in (source, Path(str(source)+'.outcome.json'))}, unix=time.time()))


def selection(metrics, order):
    tiers = ('hunter_hard', 'hunter_normal', 'hunter_easy')
    base = metrics['baseline']; detail = {}
    for label, cs in metrics.items():
        p1 = {t: cs[t]['P1']['win_minus_loss'] for t in tiers}
        p0 = sum(cs[t]['P0']['win_minus_loss'] for t in tiers)
        selfd = sum(cs[t][s]['self_deaths'] for t in tiers for s in ('P0', 'P1'))
        guards = dict(each_P1=all(p1[t] >= base[t]['P1']['win_minus_loss']-5 for t in tiers),
            P0=p0 >= sum(base[t]['P0']['win_minus_loss'] for t in tiers)-10,
            self_deaths=selfd <= sum(base[t][s]['self_deaths'] for t in tiers for s in ('P0', 'P1')))
        detail[label] = dict(P1=p1, P1_sum=sum(p1.values()), P0_sum=p0, self_deaths=selfd,
                             guards=guards, eligible=all(guards.values()))
    winner = 'baseline'
    for label in order:
        if detail[label]['eligible'] and detail[label]['P1_sum'] > detail[winner]['P1_sum']:
            winner = label
    return dict(winner=winner, comparison=detail, replacement_suggested=winner != 'baseline',
                publication='PROHIBITED_PENDING_ORCHESTRATOR_CONFIRMATION')


def aggregate(p):
    raw = {}; files = {}; reused = 0
    for j in jobs(p):
        rows = accepted(j, p)
        assert rows is not None
        a = read(str(target(j))+'.ACCEPTED.json'); files.update(a['files'])
        reused += len(rows) if a['reused'] else 0
        raw.setdefault(j['kind'], {}).setdefault(j['label'], {}).setdefault(j['condition'], {}).setdefault(j['mode'], []).extend(rows)
    metrics = {}
    for kind, labels in raw.items():
        metrics[kind] = {}
        for label, cs in labels.items():
            metrics[kind][label] = {}
            for c, modes in cs.items():
                metrics[kind][label][c] = {}
                for mode, rows in modes.items():
                    assert len(rows) == 256 == len({(r['seed'], r['seat']) for r in rows})
                    assert [sum(r['seat'] == s for r in rows) for s in (0, 1)] == [128, 128]
                    metrics[kind][label][c][mode] = {f'P{s}': summarize(rows, s) for s in (0, 1)}
    fresh = {l: {c: z['greedy'] for c, z in cs.items()} for l, cs in metrics['fresh'].items()}
    result = dict(status='COMPLETE', metrics=metrics, files=files, games=2816, reused_games=reused,
        new_games=2816-reused, selection=selection(fresh, p['candidate_order']),
        paired_per_game={l: {c: paired(z['greedy'], raw['fresh']['baseline'][c]['greedy']) for c, z in cs.items()}
                         for l, cs in raw['fresh'].items() if l != 'baseline'},
        holdout_matched_prefix={m: {f'P{s}': summarize(es[:128], s) for s in (0, 1)}
                                for m, es in raw['holdout']['baseline']['holdout'].items()},
        protocol_sha256=sha(OUT/'PROTOCOL.json'), unix=time.time())
    atomic_json(OUT/'RESULTS.json', result)
    write_report(result)
    return result


def write_report(r):
    lines = ['# R3 本侧出生补测与配对复核（20261031）', '',
        '生产JS bun-native-v2、原生本侧出生/复活、老师关闭、整局终局≤2400tick。三档fresh各256局（每席128），argmax；u3040既有768局只复算、不重跑。',
        'u3040 holdout seed2026317200：argmax与categorical分别256局，每模式每席128；原协议共同前128局单列。不能把两种动作模式相加当成独立地图或单模式128局。', '',
        '|模型|档位|P0胜/平/负|P0胜负差|P0击杀/局|P1胜/平/负|P1胜负差|P1击杀/局|',
        '|---|---|---|---:|---:|---|---:|---:|']
    for label, cs in r['metrics']['fresh'].items():
        for c, modes in cs.items():
            vs = [modes['greedy'][s] for s in ('P0', 'P1')]
            cells = []
            for v in vs:
                cells.extend([f"{v['wins']}/{v['draws']}/{v['losses']}", f"{v['win_minus_loss']:+d}", f"{v['causal_kills']/v['games']:.4f}"])
            lines.append('|'+ '|'.join([label, c]+cells)+'|')
    lines += ['', '|u3040 holdout模式|席位|局数|胜/平/负|胜负差|击杀/局|', '|---|---|---:|---|---:|---:|']
    for mode, seats in r['metrics']['holdout']['baseline']['holdout'].items():
        for seat, v in seats.items():
            lines.append(f"|{mode}|{seat}|{v['games']}|{v['wins']}/{v['draws']}/{v['losses']}|{v['win_minus_loss']:+d}|{v['causal_kills']/v['games']:.4f}|")
    lines += ['', '## 预声明选型判定', '',
        '|模型|P1三档合计|P0三档合计|每档P1≥基线−5|P0≥基线−10|自伤不增|', '|---|---:|---:|---|---|---|']
    for label, v in r['selection']['comparison'].items():
        g = v['guards']; lines.append(f"|{label}|{v['P1_sum']:+d}|{v['P0_sum']:+d}|{g['each_P1']}|{g['P0']}|{g['self_deaths']}|")
    lines += ['', f"按冻结规则建议：{r['selection']['winner']}。严格P1合计改善且所有护栏通过才建议更换；平手保留网页u3040。此任务仅报告，等待编排确认，不发布。", '',
        '逐局配对差、原始轨迹文件SHA、所有分母和完整分项指标见 `runs/r3_ownside_recheck_1620_20261031/RESULTS.json`。',
        '限制：固定单图16个重复出生组合，fresh清单已用于先前选型，并非全新独立保留集；Sim.step不代表真人frameStep。因果击杀来自引擎归因，但观测性胶/香蕉后续事件不证明干预收益。',
        'CPU评估不写训练控制、奖励、BEST或网页；持续R3-sp及原训练均不依赖本评估结果。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true'); args = parser.parse_args()
    p = freeze()
    if args.freeze:
        return
    guard = lock(OUT/'OWNER.lock')
    assert all(sha(Path(f)) == h for f, h in p['source_sha256'].items())
    if (OUT/'COMPLETE.json').exists():
        return
    pending = [j for j in jobs(p) if accepted(j, p) is None]
    runtime = (read(OUT/'CPU_SCHEDULING_ACTIVE.json') if (OUT/'CPU_SCHEDULING_ACTIVE.json').exists() else
               read(OUT/'SCHEDULING_AMENDMENT01.json') if (OUT/'SCHEDULING_AMENDMENT01.json').exists() else p)
    assert runtime['workers'] <= len(runtime['cpu_cores'])
    started = time.time()
    def progress():
        proofs = [read(f) for f in (OUT/'shards').glob('*/*/*/*/*.ACCEPTED.json')]
        return len(proofs)*8, sum(not r['reused'] for r in proofs)*8
    initial_new = progress()[1]
    with ThreadPoolExecutor(max_workers=runtime['workers']) as pool:
        active = {}
        while pending or active:
            for future in list(active):
                if future.done():
                    future.result(); del active[future]
            limit = min(runtime['workers'], read(OUT/'CPU_WORKER_LIMIT.json')['workers']) if (OUT/'CPU_WORKER_LIMIT.json').exists() else runtime['workers']
            for core in runtime['cpu_cores']:
                if not pending:
                    break
                if len(active) >= limit:
                    break
                if core in active.values():
                    continue
                j = pending.pop(0); active[pool.submit(collect, j, p, core)] = core
            count, new_count = progress()
            atomic_json(OUT/'EXECUTION_STATUS.json', dict(status='RUNNING_CPU_ONLY', completed_games=count,
                completed_new_games=new_count, completed_reused_games=count-new_count,
                budget=2816, active_workers=len(active), pending_shards=len(pending),
                new_games_per_minute_since_start=(new_count-initial_new)/max(time.time()-started, 1)*60,
                cpu_cores=runtime['cpu_cores'], worker_limit=limit, training_untouched=True, no_publish=True, unix=time.time()))
            time.sleep(3)
    result = aggregate(p)
    atomic_json(OUT/'COMPLETE.json', dict(status='COMPLETE', games=result['games'], reused_games=result['reused_games'],
        results_sha256=sha(OUT/'RESULTS.json'), report_sha256=sha(REPORT), selection=result['selection'], unix=time.time()))
    with Path('docs/training_experiment_ledger.md').open('a') as f:
        f.write(f"\nR3-OWNSIDE-RECHECK-1620-20261031终态COMPLETE：2816完整局，复用{result['reused_games']}、新增{result['new_games']}；建议{result['selection']['winner']}，不发布。报告 `{REPORT}`；结果SHA `{sha(OUT/'RESULTS.json')}`。\n")
    message = f"R3本侧复核完成：2816完整局，u3040 holdout双模式每席128；三档建议{result['selection']['winner']}。报告{REPORT}；不发布，等待编排确认，训练继续。"
    notification = subprocess.run(['jdme-notify', 'send', message], capture_output=True, text=True)
    atomic_json(OUT/'NOTIFIED.json', dict(returncode=notification.returncode, message=message, unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('existing evaluator owns this namespace')
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT/'BLOCKED.json', dict(status='BLOCKED', error_type=type(error).__name__, reason=str(error),
            training_untouched=True, unix=time.time()))
        raise
