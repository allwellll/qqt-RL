"""Prepare only the winner of the immutable paired selection; no network writes."""
import copy
import hashlib
import json
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, sha

B = Path('runs/bomb_threat_v1')
O = B / 'publish_best_20261030'
R = B / 'replay_best_slot00_u2400_20261030'
ID = 'bun392-a3sr-slot00-u2400'
TAG = 'web-bun392-a3sr-slot00-u2400-20261030'


def read(p):
    return json.loads(Path(p).read_text())


def main():
    global ID, TAG, R
    decision = read(O / 'FINAL_SELECTION.json')
    assert read(O / 'FINAL_SELECTION_COMPLETE.json')['selection_sha256'] == sha(O / 'FINAL_SELECTION.json')
    if not decision['publish']:
        atomic_json(O / 'PUBLICATION_STATUS.json', dict(status='RETAINED_BASELINE', selection_sha256=sha(O / 'FINAL_SELECTION.json')))
        return
    cfg = read(O / 'SELECTION_PROTOCOL.json')
    role = decision['winner']
    source = decision['models'][role]
    slot = source['slot']
    ID = f'bun392-a3sr-slot{slot:02d}-u2400'
    TAG = f'web-{ID}-20261030'
    R = B / f'replay_best_slot{slot:02d}_u2400_20261030'
    cp = Path(source['checkpoint'])
    cm = read(cp / 'COMMITTED.json')
    assert sha(cp / 'policy.pt') == cm['policy_sha256'] == source['policy_sha256']
    original = Path(source['model_json'])
    assert sha(original) == source['model_sha256']
    doc = read(original)
    graph = original.with_name('model.onnx')
    assert sha(graph) == doc['meta']['onnx_sha256']
    assert doc['meta']['checkpoint_sha256'] == source['policy_sha256']
    doc['meta'].pop('onnx_path', None)
    doc['meta'].update(name=f'Bun392 A3SR slot{slot:02d} u2400', display_name=f'Bun392 A3SR slot{slot:02d} u2400',
                       source=str(cp / 'policy.pt'), committed_sha256=sha(cp / 'COMMITTED.json'),
                       strategy='raw', state_dim=24, attention_heads=4, update=2400,
                       generated_at='2026-10-30T00:00:00Z', rule_version='stock-window-v1')
    data = (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()
    h = hashlib.sha256(data).hexdigest()
    gh = sha(graph)
    name = f'{ID}-{h}.json'
    gname = f'{ID}-{gh}.onnx'
    (O / name).write_bytes(data)
    (O / gname).write_bytes(graph.read_bytes())
    scores = decision['comparison']
    metrics = decision['results']
    summary = '；'.join(f"{t} P1 {scores[role]['P1'][t]:+d}（u1280 {scores['baseline']['P1'][t]:+d}）" for t in ('hard', 'normal', 'easy'))
    url = f'https://github.com/allwellll/qqt-RL/releases/download/{TAG}/'
    row = dict(id=ID, display_name=f'Bun392 A3SR slot{slot:02d} u2400（P1三档胜负差 {scores[role]["P1_sum"]:+d}）',
               file=name, sha256=h, bytes=len(data), url=url + name, cycle=2400,
               candidate=cp.parent.name+'_u002400_raw', source_sha256=source['policy_sha256'],
               strategy='raw', score=metrics[role]['hard']['win'] / 256,
               description=f'网页真人坐P0、模型坐P1；生产JS猎手bun.hunter，stock-window-v1，argmax，老师关闭。预声明同seed配对每档256局（P0/P1各128）：{summary}；P1合计{scores[role]["P1_sum"]:+d} vs u1280 {scores["baseline"]["P1_sum"]:+d}，P0合计{scores[role]["P0_sum"]:+d} vs {scores["baseline"]["P0_sum"]:+d}。通过每档/P0/自伤护栏。已知短板：零泡局、振荡、持包受砖阻；不宣称全部战斗指标或真人泛化提升。训练bf16与网页fp32存在数值差异。',
               score_protocol='production JS bun.hunter; stock-window-v1; argmax; teacher off; 256 paired games/tier, P0/P1 128 each; seeds hard2026193000 normal2026194000 easy2026195000; score=hard wins/256; 64 geometries reused; no significance claim',
               onnx=dict(file=gname, sha256=gh, bytes=graph.stat().st_size, url=url + gname, meta=doc['meta']))
    atomic_json(O / 'CATALOG_ROW.json', row)
    atomic_json(O / 'manifest.json', dict(schema='qqt.bun-release/v1', tag=TAG, source_ref=str(cp),
                policy_sha256=source['policy_sha256'], committed_sha256=sha(cp / 'COMMITTED.json'),
                rule_version='stock-window-v1', strategy='raw', selection_sha256=sha(O / 'FINAL_SELECTION.json'),
                raw_tensors_unchanged=doc['flat'] == read(original)['flat'] and doc['tensors'] == read(original)['tensors'],
                comparison=scores, assets=[dict(file=name, sha256=h, bytes=len(data)), dict(file=gname, sha256=gh, bytes=graph.stat().st_size)]))
    (O / 'SHA256SUMS').write_text(f'{h}  {name}\n{gh}  {gname}\n{sha(O / "manifest.json")}  manifest.json\n')
    # The replay panel intentionally reuses the old fixed hard seed list for
    # user-visible comparisons; it is not substituted into release selection.
    watch = B / 'iter2/a3sr/lineage_watch_20261030'
    oldproto = read(watch / 'PROTOCOL.json')
    rows = []
    primary = {}
    files = {}
    for start in range(0, 256, 8):
        p = watch / f'fresh/slot{slot:02d}_u002400/hard' / f'greedy_{start:03d}.json'
        proof = read(Path(str(p) + '.ACCEPTED.json'))
        for f, digest in proof['files'].items():
            assert sha(f) == digest
            files[f] = digest
        rows += read(Path(str(p) + '.outcome.json'))['episodes']
        primary.update({e['seed']: e for e in read(p)['results']['natural_full2400']['episodes']})
    rows.sort(key=lambda r: r['seed'])
    assert len(rows) == len({r['seed'] for r in rows}) == 256
    games, missing = [], []

    def add(r, rule):
        r = copy.deepcopy(r)
        r['primary_delivery'] = primary[r['seed']]['delivery']
        games.append(dict(index=r['seed'] - oldproto['fresh']['tiers']['hard']['seed'],
                          seed=r['seed'], seat=r['seat'], rule=rule, expected=r))

    for r in rows[:8]:
        add(r, '固定首8局；与u800/u1280页同seed')
    for seat in (0, 1):
        wins = [r for r in rows if r['seat'] == seat and r['winner'] == seat][:3]
        losses = [r for r in rows if r['seat'] == seat and r['winner'] not in (None, -1, seat)][:1]
        for r in wins:
            add(r, f'P{seat}按index最早3胜')
        for r in losses:
            add(r, f'P{seat}按index最早1负')
        if len(wins) < 3 or not losses:
            missing.append(f'P{seat}: wins available {len(wins)}, losses available {len(losses)}')
    zero = [r for r in rows if r['seat'] == 1 and r['placed'][1] == 0 and r['placed'][0] > 0][:2]
    for r in zero:
        add(r, 'P1按index最早2个单方零泡')
    if len(zero) < 2:
        missing.append(f'P1 unilateral zero only {len(zero)}')
    for g in read(B / 'iter2/a3sr/BRICK_OBSERVATION_MANIFEST.json')['games'][:3]:
        add(next(r for r in rows if r['seed'] == g['seed'] and r['seat'] == g['seat']), 'u800砖隔开清单前3个同seed')
    R.mkdir(parents=True, exist_ok=True)
    atomic_json(R / 'MANIFEST.json', dict(status='FROZEN_BEFORE_RECORDING', model=source, games=games,
                missing=missing, source_files=files, spawn_manifest=oldproto['fresh']['tiers']['hard']['spawn_manifest'],
                selection_instruction='.hermes/TASK_best_publish_and_cleanup_20261030_1530.md',
                model_label=f'A3SR slot{slot:02d} u2400 · 网页默认候选', html=f'20261030-a3sr-slot{slot:02d}-u2400-review.html',
                teacher=False, rule_version='stock-window-v1', unix=time.time()))
    atomic_json(O / 'REPLAY_PREPARED.json', dict(base=str(R), manifest_sha256=sha(R / 'MANIFEST.json'),
                html=f'20261030-a3sr-slot{slot:02d}-u2400-review.html'))
    print(json.dumps(dict(status='PREPARED', tag=TAG, model_sha256=h, graph_sha256=gh,
                         replay_entries=len(games), unique_games=len({g['index'] for g in games}))))


if __name__ == '__main__':
    main()
