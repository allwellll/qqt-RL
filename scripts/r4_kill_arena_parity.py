"""64 complete deterministic same-action games; random item parity separate."""
import gzip
import json
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import subprocess
import time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json,lock,sha
from scripts import r4_kill_arena as K

B=Path(os.environ.get('R4_PARITY_OUT','runs/r4_combat_20261031/parity'))
FIELD_MAP={'pos':lambda a:a.state.pos,'alive':lambda a:a.state.alive,'hp':lambda a:a.state.hp,
    'invuln':lambda a:a.state.core.invuln,'bombsCap':lambda a:a.state.core.bombs_cap,
    'blastCap':lambda a:a.state.core.blast_cap,'spdG':lambda a:a.state.core.spd_g,
    'trapped':lambda a:a.state.trapped,'wall':lambda a:a.state.wall,'brick':lambda a:a.state.brick,
    'bush':lambda a:a.state.core.bush,'pushable':lambda a:a.state.core.pushable,
    'crate':lambda a:a.state.core.crate,'fuse':lambda a:a.state.fuse,'owner':lambda a:a.state.core.owner,
    'bombBlast':lambda a:a.state.core.bomb_blast,
    # Native JAX packs horizontal/vertical remaining fire ticks into nibbles.
    # Compare BOTH components and their max rather than hiding an ABI mismatch.
    'blastLinger':lambda a:j.maximum(a.state.core.blast_linger>>4,a.state.core.blast_linger&15),
    'horzLinger':lambda a:a.state.core.blast_linger>>4,'vertLinger':lambda a:a.state.core.blast_linger&15,
    'brickLinger':lambda a:a.state.core.brick_linger,'pushT':lambda a:a.state.core.push_t,
    'fieldItem':lambda a:a.state.field_item,'fieldOwner':lambda a:a.state.field_owner,
    'fieldArmed':lambda a:a.state.field_armed,'movementStatus':lambda a:a.state.move_status,
    'movementStatusTicks':lambda a:a.state.status_ticks,'slideDir':lambda a:a.state.slide_dir,
    'lastMoveDir':lambda a:a.state.last_move_dir,'arenaKills':lambda a:a.kills,'arenaDeaths':lambda a:a.deaths,
    'arenaSelf':lambda a:a.self_deaths,'arenaLifeAge':lambda a:a.life_age}

def main():
    B.mkdir(parents=True,exist_ok=True);guard=lock(B/'OWNER.lock');start=time.monotonic()
    catalog=json.loads(K.CATALOG.read_text());train=[i for i,m in enumerate(catalog['maps']) if m['split']=='train']
    manifest=B/'MANIFEST.json'
    if not manifest.exists():atomic_json(manifest,dict(status='FROZEN',games=64,horizon=1200,
        fixtures=[dict(index=i,map_index=train[i],seed=2026332000+i) for i in range(64)],
        float_atol=3e-4,discrete='exact',items='disabled',random_items='not part of deterministic gate; separate distribution evaluation required',
        source_sha256={f:sha(Path(f)) for f in ['web/sim-kill-arena.js','scripts/r4_kill_arena.py','scripts/r4_kill_arena_fixture.js']},unix=time.time()))
    p=json.loads(manifest.read_text());assert all(sha(Path(f))==h for f,h in p['source_sha256'].items())
    K.configure()
    # No random crate drops in this declared deterministic gate. Native fixed
    # pickup fixtures and independent random-drop statistics are separate.
    K._TABLE.rate=j.zeros_like(K._TABLE.rate)
    @jax.jit
    def scan(a,actions,seed):
        def body(a,data):
            act,t=data;a,done,info=K.step(a,act,jax.random.fold_in(jax.random.PRNGKey(seed),t),True)
            return a,dict({k:f(a) for k,f in FIELD_MAP.items()},done=done,winner=info['winner'])
        return jax.lax.scan(body,a,(actions,j.arange(1200)))
    results=[]
    for case in p['fixtures']:
        target=B/f"game_{case['index']:03d}.json.gz";receipt=B/f"game_{case['index']:03d}.ACCEPTED.json"
        if receipt.exists():
            r=json.loads(receipt.read_text());assert sha(target)==r['fixture_sha256'];results.append(r);continue
        if not target.exists():subprocess.run(['node','scripts/r4_kill_arena_fixture.js',str(case['map_index']),str(case['seed']),str(target)],check=True)
        d=json.loads(gzip.decompress(target.read_bytes()));a=K.template(d['seed'],d['map_index'],d['initial'])
        actions=j.array([[[x[0],2 if len(x)>2 and x[2] else x[1]] for x in row['actions']] for row in d['rows']],j.int32)
        _,actual=scan(a,actions,j.array(d['seed'],j.uint32));actual=jax.device_get(actual)
        mismatches=[];max_position=0.
        for tick,row in enumerate(d['rows']):
            fields=[]
            for name in FIELD_MAP:
                x=np.asarray(actual[name][tick]).reshape(-1);expected=np.asarray(row['after'][name]).reshape(-1)
                same=np.allclose(x,expected,rtol=0,atol=3e-4 if name in ('pos','spdG') else 1e-6) if name in ('pos','spdG','pushT') else np.array_equal(x,expected)
                if name=='pos':max_position=max(max_position,float(np.max(np.abs(x-expected))))
                if not same:fields.append(name)
            if bool(actual['done'][tick])!=row['done']:fields.append('done')
            expected_winner=row['winner'] if row['winner'] is not None else -1
            if row['done'] and int(actual['winner'][tick])!=expected_winner:fields.append('winner')
            if fields:
                first={name:dict(jax=np.asarray(actual[name][tick]).reshape(-1).tolist(),js=row['after'][name]) for name in fields if name in FIELD_MAP}
                mismatches.append(dict(tick=tick+1,fields=fields,values=first));break
        r=dict(status='PASS' if not mismatches else 'BLOCKED',case=case,ticks=1200,compared_ticks=mismatches[0]['tick'] if mismatches else 1200,
            full_native_scan_executed=True,position_max_error_through_first_divergence=max_position,first_divergence=mismatches,
            fixture=str(target),fixture_sha256=sha(target),js_final_kills=d['rows'][-1]['after']['arenaKills'],
            jax_final_kills=np.asarray(actual['arenaKills'][-1]).tolist(),unix=time.time())
        atomic_json(receipt,r);results.append(r)
        atomic_json(B/'PROGRESS.json',dict(status='RUNNING',games=len(results),passed=sum(x['status']=='PASS' for x in results),budget=64,wall_seconds=time.monotonic()-start,unix=time.time()))
    passed=sum(x['status']=='PASS' for x in results)
    atomic_json(B/'GATE.json',dict(status='PASS' if passed==64 else 'BLOCKED',games=64,passed=passed,failed=64-passed,
        records=[str(B/f"game_{i:03d}.ACCEPTED.json") for i in range(64)],manifest_sha256=sha(manifest),
        wall_seconds=time.monotonic()-start,training_authorized=passed==64,
        limitations=['only declared no-random-item Sim.step fixtures','fixed-item fixtures and random item distribution comparisons must be reported separately','not human frameStep'],unix=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        B.mkdir(parents=True,exist_ok=True)
        atomic_json(B/'TECHNICAL_BLOCKED.json',dict(status='BLOCKED',error_type=type(exc).__name__,reason=str(exc),training_signals=0,unix=time.time()));raise
