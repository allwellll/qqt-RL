"""Read-only episode-level audit of the already accepted JS outcomes."""
import json
import time
from pathlib import Path
from scripts.population_run_state import atomic_json, sha

SOURCE = Path('runs/r2a_yield_r3b_verify_20261031/evaluation/r3_a_u3600')
OUT = Path('runs/r3_selfplay_ramp_20261031/audit/guard1600')


def summarize(episodes):
    groups = {key:dict(games=0,wins=0,draws=0,losses=0,deaths=0,self_deaths=0,contacts=0)
              for key in ('carrying_own_glue','no_carrying_own_glue')}
    rows = []
    selected = [episode for episode in episodes if episode['seat'] == 0]
    assert len({e['seed'] for e in selected}) == len(selected), 'duplicate episode'
    for e in selected:
        contacts = [c for c in e['r3']['contacts']
                    if c['type'] == 2 and c['owner'] == 0 and c['player'] == 0 and c['carried'] >= 0]
        key = 'carrying_own_glue' if contacts else 'no_carrying_own_glue'
        result = 'wins' if e['winner'] == 0 else 'draws' if e['winner'] in (-1,None) else 'losses'
        assert e['winner'] in (-1,None,0,1)
        row = groups[key]; row['games'] += 1; row[result] += 1
        row['deaths'] += e['deaths'][0]; row['self_deaths'] += e['self_deaths'][0]
        row['contacts'] += len(contacts)
        rows.append(dict(seed=e['seed'],seat=0,group=key,result=result,
                         winner=e['winner'],deaths=e['deaths'][0],self_deaths=e['self_deaths'][0],
                         contacts=len(contacts),contact_ticks=[c['tick'] for c in contacts]))
    for row in groups.values():
        assert row['wins']+row['draws']+row['losses'] == row['games']
        row['deaths_per_game'] = row['deaths']/row['games'] if row['games'] else None
    return groups, rows


def main():
    complete = json.loads((SOURCE/'COMPLETE.json').read_text())
    episodes = []; hashes = {}
    for start in range(0,128,8):
        path = SOURCE/'hunter_hard'/f'shard_{start:03d}.json.outcome.json'
        digest = sha(path)
        assert complete['files'][str(path)] == digest, 'accepted input changed'
        value = json.loads(path.read_text())
        assert value['mode'] == 'greedy' and not value['teacher']
        episodes.extend(value['episodes']); hashes[str(path)] = digest
    assert len(episodes) == 128 and len({(e['seed'],e['seat']) for e in episodes}) == 128
    groups, rows = summarize(episodes)
    assert len(rows) == 64
    assert {r['seed'] for r in rows} == set(range(2026334500,2026334628,2))
    totals = {key:sum(g[key] for g in groups.values())
              for key in ('games','wins','draws','losses','deaths','self_deaths','contacts')}
    assert (totals['wins'],totals['draws'],totals['losses'],totals['contacts']) == (59,3,2,66)
    result = dict(status='COMPLETE',model='r3_a_u3600',condition='hunter_hard',seat=0,
                  rule='bun-native-v2',spawn_side='own-side-native-v1',groups=groups,totals=totals,
                  episodes=rows,input_sha256=hashes,accepted_complete_sha256=sha(SOURCE/'COMPLETE.json'),
                  causal=False,limitations=['observational association only; no paired intervention',
                    'longer carrying exposure may increase contact opportunity',
                    'deaths are whole-game totals, not deaths attributed to glue',
                    'single map and fixed natural seed set; no reward change'],unix=time.time())
    atomic_json(OUT/'OWN_GLUE_AUDIT.json',result)
    print(json.dumps(dict(groups=groups,totals=totals),ensure_ascii=False))


if __name__ == '__main__':
    main()
