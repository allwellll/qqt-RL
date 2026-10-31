"""Authorized A3SR640 audit, fresh recheck, and bounded full-state successors."""
import json
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bomb_threat_v1'
OUT = BASE / 'iter2/a3sr/round14'
AUTH = OUT / 'AUTHORIZATION.json'
DONOR = 'bt_slot05_c000026'
INSTRUCTION = '.hermes/STEER_a3sr_fresh256_20261030_0405.md'


def read(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def audit():
    from scripts.bomb_threat_gate_report import recompute_natural, reward_monitor
    from scripts.bomb_threat_gates_v2 import evaluate
    out = BASE / 'eval' / (DONOR + '_u000640_holdout')
    marker, natural = recompute_natural(out)
    _, parent = recompute_natural(BASE / 'eval/baseline_parent')
    refs = {n: recompute_natural(BASE/'eval'/n)[1] for n in
            ('reference_transport1536', 'reference_selfplay3072')}
    reward = reward_monitor(ROOT / marker['checkpoint'], 640)
    gates = evaluate(natural, parent, reward, 'holdout', refs)
    formal = read(out / 'GATE_METRICS.json')
    assert gates['absolute']['status'] == formal['absolute_gate_status'] == 'PASS'
    assert gates['relative']['status'] == formal['pbt_gate_status'] == 'PASS'
    expected = dict(win=26, draw=38, loss=0, causal_kills=45, deaths=4,
                    actual_bombs=2881, zero_bomb_games=3, draw_kill_difference=27,
                    self_deaths=0, cage_kills=0)
    measured = natural['greedy']
    assert all(measured[k] == v for k, v in expected.items())
    rows = []
    files = {}
    for p in sorted(out.glob('natural_greedy_[0-9][0-9][0-9].json.outcome.json')):
        files[str(p.relative_to(ROOT))] = sha(p)
        for r in read(p)['episodes']:
            s = r['seat']
            rows.append(dict(seed=r['seed'], seat=s, ticks=r['ticks'],
                             placed=r['placed'][s], requests=r['requests'][s],
                             causal_kills=r['causal_kills'][s], deaths=r['deaths'][s],
                             winner=r['winner']))
    assert len(rows) == len({r['seed'] for r in rows}) == 64
    def stats(rr):
        values = sorted(r['placed'] for r in rr)
        return dict(games=len(rr), bombs=sum(values), mean=sum(values)/len(rr) if rr else None,
                    min=values[0] if rr else None, max=values[-1] if rr else None,
                    sorted_bombs=values, causal_kills=sum(r['causal_kills'] for r in rr))
    result = dict(status='PASS', gates=gates, orchestrator_differences=[],
                  natural=natural, reward_monitor=reward, episodes=rows,
                  groups={k:stats(rr) for k,rr in {
                      'all':rows, 'P0':[r for r in rows if r['seat']==0],
                      'P1':[r for r in rows if r['seat']==1],
                      'kill_games':[r for r in rows if r['causal_kills']>0],
                      'nonkill_games':[r for r in rows if r['causal_kills']==0]}.items()},
                  causal_kills_per_placed_bomb=45/2881,
                  hit_rate=None, missing=['per-bomb hit attribution absent in immutable outcome shards'],
                  limitations=['kills/bombs is an aggregate event ratio, not per-bomb hit probability or causal efficacy',
                               'high bomb volume alone neither proves nor rules out farming'],
                  files=files, gate_metrics_sha256=sha(out/'GATE_METRICS.json'), unix=time.time())
    atomic_json(OUT/'HOLDOUT_AUDIT.json', result)
    return result


def freeze():
    result = audit()
    done = read(BASE/'formal'/DONOR/'COMPLETE.json')
    for file,key in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
        assert sha(ROOT/done['checkpoint']/file) == done[key]
    path = AUTH
    if not path.exists():
        slots = read(BASE/'SLOTS.json')
        atomic_json(path, dict(status='AUTHORIZED', instruction=INSTRUCTION,
            instruction_sha256=sha(ROOT/INSTRUCTION), donor=done,
            slot05_end_update=1280, slot07_boundary_run=slots['7']['run'],
            slot07_end_update=1280, copy_count=1, seed_by_slot={'5':2026112905,'7':2026112907},
            copied=['Actor','Critic','Adam','env','aux','RNG','ring','ring_cursor','fixed_actor','counters'],
            slot07_rng='donor parity first, then target seed rekey; donor environment/ring preserved',
            no_promotion=True, no_publication=True, unix=time.time()))
    atomic_json(OUT/'TASK_CONSUMED.json', dict(status='AUDIT_PASSED_SUCCESSORS_AUTHORIZED',
        authorization=str(AUTH.relative_to(ROOT)), authorization_sha256=sha(AUTH),
        audit_sha256=sha(OUT/'HOLDOUT_AUDIT.json'), unix=time.time()))
    return result


def make_protocol(slot, chunk, source, recipe, length=320, rekey=False):
    auth = read(AUTH, {})
    assert auth.get('status') == 'AUTHORIZED' and slot in (5,7)
    assert recipe['iter2_arm'] == 'A3SR' and 640 <= source['update'] < 1280
    length = min(length, 1280-source['update'])
    donor_proto = read(BASE/'protocols'/f'{DONOR}.json')
    assert recipe == donor_proto['arms'][DONOR]
    name = f'bt_slot{slot:02d}_c{chunk:06d}'
    path = BASE/'protocols'/f'{name}.json'
    if path.exists():
        p = read(path)
        assert p['continuation_source'] == source and p['arms'][name] == recipe
        return path
    proto = dict(donor_proto)
    hashes = dict(donor_proto['source_sha256'])
    for f in ('scripts/bomb_threat_round14.py','scripts/bomb_threat_a3sr_resume_train.py'):
        hashes[f] = sha(ROOT/f)
    assert all(sha(ROOT/f)==h for f,h in hashes.items()), 'frozen A3SR worker ABI changed'
    seed = auth['seed_by_slot'][str(slot)]
    proto.update(namespace_id='A3SR-round14', arms={name:recipe}, sources={str(seed):source},
        continuation_source=source, chunk_start_update=source['update'], chunk_updates=length,
        environment_steps_per_run=(source['update']+length)*262144, slot=slot, chunk=chunk,
        source_sha256=hashes, worker_module='scripts.bomb_threat_a3sr_resume_train',
        round14_authorization=str(AUTH.relative_to(ROOT)), round14_collector_seed=2026112905,
        round14_rekey=rekey, initialization='complete state continuation; no Critic warmup or reward/ledger reset',
        pbt_lineage=dict(donor=source['checkpoint'], state_sha256=source['state_sha256'],
            policy_sha256=source['policy_sha256'], target_slot=slot, target_seed=seed,
            rng='independent target seed after donor parity' if rekey else 'continued exactly',
            critic_warmup_updates=0, formal_updates_discarded=0))
    atomic_json(path,proto)
    return path


def apply_switch(slot, entry, done, chunk):
    auth = read(AUTH,{})
    if auth.get('status')!='AUTHORIZED' or slot not in (5,7) or entry.get('round14_applied'):
        return None
    boundary = DONOR if slot==5 else auth['slot07_boundary_run']
    if entry['run'] != boundary:
        return None
    assert read(BASE/'formal'/boundary/'COMPLETE.json')['state_sha256']==done['state_sha256']
    source = auth['donor']
    for f,k in [('state.pkl','state_sha256'),('policy.pt','policy_sha256')]:
        assert sha(ROOT/source['checkpoint']/f)==source[k]
    recipe = read(BASE/'protocols'/f'{DONOR}.json')['arms'][DONOR]
    path = make_protocol(slot,chunk,source,recipe,rekey=slot==7)
    receipt = dict(status='PROTOCOL_READY',protocol=str(path.relative_to(ROOT)),
        protocol_sha256=sha(path),source=source,old_last_atomic=done,
        old_state_retained=True,training_worker_signals=0,unix=time.time())
    atomic_json(OUT/f'SLOT{slot:02d}_HANDOFF.json',receipt)
    atomic_json(BASE/'formal'/boundary/'ROUND14_SUCCESSOR.json',receipt)
    return dict(entry, run=path.stem,chunk=chunk,source=source,iter2_arm='A3SR',
        role='A3SR_round14_continuation' if slot==5 else 'A3SR_round14_PBT_copy',
        round14_applied=True, supersedes=boundary)


def terminal(entry, done):
    if not entry.get('round14_applied') or done['update']<1280:
        return False
    assert done['update']==1280
    path=OUT/f"SLOT{entry['gpu']:02d}_TERMINAL.json"
    if not path.exists():
        receipt=dict(status='FIXED_AUTHORIZED_BUDGET_COMPLETE',run=entry['run'],last_atomic=done,
                     next='await explicit authorization for A3SR extension',unix=time.time())
        atomic_json(path,receipt)
        atomic_json(BASE/'notifications'/path.name,receipt)
        from scripts.iter_pbt_owner import notify
        notify(path.stem,f"A3SR round14 {entry['run']} u1280完整原子保存，固定预算结束；其他训练继续，未晋升。")
    return True


def refresh():
    for slot in (5,7):
        handoff=read(OUT/f'SLOT{slot:02d}_HANDOFF.json',{})
        accepted=OUT/f'SLOT{slot:02d}_FIRST_FORMAL.json'
        if not handoff or accepted.exists():continue
        name=Path(handoff['protocol']).stem; out=BASE/'formal'/name
        smoke=read(out/'SMOKE.json',{});restore=read(out/'RESTORE.json',{})
        if not (smoke.get('passed') and restore.get('passed')):continue
        rows=[]
        for line in (out/'updates.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
        formal=[r for r in rows if r['new_update']>642]
        if not formal:continue
        r=formal[0];assert r['finite'] and min(r['update_l2'].values())>0
        receipt=dict(status='PASS',run=name,update=r['new_update'],finite=r['finite'],
            update_l2=r['update_l2'],environment_steps=r['environment_steps'],
            env_per_second=262144/r['cycle_seconds'],smoke=smoke,restore=restore,
            handoff=read(out/'SOURCE_HANDOFF.json'),unix=time.time())
        atomic_json(accepted,receipt);atomic_json(BASE/'notifications'/accepted.name,receipt)
        from scripts.iter_pbt_owner import notify
        notify(accepted.stem,f"A3SR round14 slot{slot:02d}正式u{r['new_update']}，finite、Actor/Critic非零，{receipt['env_per_second']:.0f}env/s；完整恢复通过，未晋升。")


if __name__=='__main__':
    print(json.dumps(freeze()['gates']))
