"""Per-tick native custody events on complete, same-action diagnostic games."""
import json
import os
import subprocess
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from jax_bomb import je07_bun_env_r1_items as E,je07_native_items as I
from scripts.population_run_state import atomic_json,sha

OUT=Path('runs/bun_native_v2_20261030/parity/event_v3')


def fresh(doc):
    E.prepare();E.configure_training('full=1',5,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=0)
    E._DANGER_TRACKING=False
    s=jax.tree.map(lambda x:x[0],E.init_batch(jax.random.PRNGKey(doc['seed']),1,jax.nn.one_hot(0,16)))
    d=doc['initial'];z=j.zeros_like(s.wall)
    c=s.core._replace(wall=j.array(d['wall'],bool).reshape(13,15),brick=z,bush=z,pushable=z,crate=j.zeros_like(s.core.crate),
        fuse=j.zeros_like(s.fuse),owner=j.full_like(s.core.owner,-1),blast_linger=j.zeros_like(s.core.blast_linger),
        pos=j.array(d['pos'],j.float32).reshape(2,2),invuln=j.zeros_like(s.core.invuln),hp=j.array(d['hp'],s.hp.dtype),
        bombs_cap=j.array(d['stats'][0],s.core.bombs_cap.dtype),blast_cap=j.array(d['stats'][1],s.core.blast_cap.dtype),spd_g=j.array(d['stats'][2],j.float32))
    return s._replace(core=c,scene=j.array(8,j.int32),lesson=j.array(E.LESSON_FULL,s.lesson.dtype),
        tactical_crate=j.zeros_like(s.tactical_crate),field_item=j.zeros_like(s.field_item),native_items=I.fresh(),
        bun_spawn_pos=j.array(d['spawn'],j.float32),trapped=j.zeros_like(s.trapped),item_slots=j.zeros_like(s.item_slots),
        item_counts=j.zeros_like(s.item_counts),held_item=j.zeros_like(s.held_item))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];failures=[]
    fn=jax.jit(lambda s,a,k:E.step(s,a,k,auto_reset=False,return_info=True,require_stock_window=False))
    for seat in (0,1):
        for i,mode in enumerate(('delivery','drop','transfer')):
            file=OUT/f'P{seat}_{mode}.json'
            if not file.exists():
                raw=subprocess.check_output(['node','scripts/bun_native_v2_parity_fixture.js',str(seat),str(2026206000+seat*10+i),mode])
                file.write_bytes(raw)
            doc=json.loads(file.read_text());s=fresh(doc);events=[];previous=np.array(s.bun_carried)
            for tick,row in enumerate(doc['rows']):
                injection=row['injection']
                if injection:s=s._replace(trapped=j.array(injection['trapped'],s.trapped.dtype),core=s.core._replace(pos=j.array(injection['pos'],j.float32).reshape(2,2)))
                s,done,info=fn(s,j.array(row['actions']),jax.random.PRNGKey(doc['seed']+tick))
                want=row['after']
                fields=dict(carried=s.bun_carried,stored=s.bun_stored,loose=s.bun_loose.reshape(-1),alive=s.alive,respawn=s.bun_respawn,trapped=s.trapped)
                mismatch=[]
                for k,v in fields.items():
                    if not np.array_equal(np.asarray(v),want[k]):mismatch.append(k)
                if bool(done)!=want['done']:mismatch.append('done')
                if want['done'] and int(info['winner'])!=want['winner']:mismatch.append('winner')
                expected_transfer=[False,False]
                for event in row.get('transfers',[]):expected_transfer[event['receiver']]=True
                if not np.array_equal(np.asarray(info['native_transfer']),expected_transfer):mismatch.append('native_transfer')
                pos_error=float(np.max(np.abs(np.asarray(s.pos).reshape(-1)-want['pos'])))
                if pos_error>3e-4:mismatch.append('position')
                if mismatch:
                    failures.append(dict(seat=seat,mode=mode,tick=tick+1,fields=mismatch,position_max_error=pos_error,
                        jax_carried=np.asarray(s.bun_carried).tolist(),js_carried=want['carried'],jax_done=bool(done),js_done=want['done']))
                    break
                if not np.array_equal(previous,np.asarray(s.bun_carried)) or np.asarray(info['death']).any():
                    events.append(dict(tick=tick+1,carried=np.asarray(s.bun_carried).tolist(),death=np.asarray(info['death']).tolist(),steal=np.asarray(info['steal']).tolist(),capture=np.asarray(info['capture']).tolist(),native_transfer=np.asarray(info['native_transfer']).tolist()))
                previous=np.array(s.bun_carried)
            records.append(dict(seat=seat,mode=mode,ticks=len(doc['rows']),compared_ticks=tick+1,done=doc['rows'][-1]['after']['done'],
                fixture=str(file),sha256=sha(file),events=events))
    coverage={f'P{seat}':dict(stock=any(any(e['steal']) for r in records if r['seat']==seat for e in r['events']),
        delivery=any(any(e['capture']) for r in records if r['seat']==seat for e in r['events']),
        death_drop=any(any(e['death']) for r in records if r['seat']==seat and r['mode']=='drop' for e in r['events']),
        contact_transfer=any(any(e['native_transfer']) for r in records if r['seat']==seat and r['mode']=='transfer' for e in r['events'])) for seat in (0,1)}
    ok=not failures and all(x['done'] for x in records) and all(all(v.values()) for v in coverage.values())
    atomic_json(OUT/'ACCEPTANCE.json',dict(status='PASS' if ok else 'BLOCKED',ruleset='bun-native-v2',records=records,failures=failures,
        coverage=coverage,
        limitations=['map806 cleared-brick deterministic action games; no assertion for untested random Hunter trajectories',
                     'drop/transfer control fixture injected after real stock acquisition, same intervention in both engines',
                     'Sim.step only; human frameStep excluded; float tolerance3e-4, custody events exact']))
    print(json.dumps(dict(status='PASS' if ok else 'BLOCKED',failures=failures)))


if __name__=='__main__':run()
