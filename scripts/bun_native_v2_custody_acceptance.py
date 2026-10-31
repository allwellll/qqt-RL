"""Real same-action JS/JAX custody paths and full composed reward audit."""
import json
import subprocess
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json,sha

OUT=Path('runs/bun_native_v2_20261030/custody_v2_tests')


def run():
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts import carrier_mechanism_engine as E
    E.configure()
    from scripts import bun_native_v2_custody_reward as R,je04_bun_ops as L,bun_native_v2_custody_collector as M
    M.configure(True,True)
    from scripts.bun_native_v2_parity import fresh
    records=[];failures=[]
    transition=jax.jit(M.transition)
    for seat in (0,1):
        for mode in ('dropcycles','cycles','delivery','loss'):
            f=OUT/f'P{seat}_{mode}.json';f.parent.mkdir(parents=True,exist_ok=True)
            if not f.exists():f.write_bytes(subprocess.check_output(['node','scripts/bun_native_v2_custody_fixture.js',str(seat),mode]))
            doc=json.loads(f.read_text());single=fresh(doc)
            s=jax.tree.map(lambda x:x[None],single);aux=E.ops.reward_state(1)
            aux=dict(aux,tactics=M._module.original.T.events.state(1),provenance_source=j.full(1,-1,j.int32),
                r1_attribute_highwater=j.zeros((1,2,3)),a3sr={v:j.zeros(1) for v in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
            aux=dict(aux,combat_mix=M.metadata(s,j.array([0]),j.array([False]),j.array([seat])))
            sums=np.zeros(2);custody=np.zeros(2);terminal=np.zeros(2);components=None;component_names=None
            rowlog=[];restores=[];mismatch=[];interceptions=[];cycle_start=None;cycle_raw=0.
            for tick,row in enumerate(doc['rows']):
                inj=row['injection']
                if inj:
                    if 'trapped' in inj:s=s._replace(trapped=j.array([inj['trapped']],s.trapped.dtype),core=s.core._replace(pos=j.array([inj['pos']],j.float32).reshape(1,2,2)))
                    if 'fuse' in inj:s=s._replace(core=s.core._replace(fuse=j.array([inj['fuse']],s.fuse.dtype).reshape(1,13,15),owner=j.array([inj['owner']],s.core.owner.dtype).reshape(1,13,15),bomb_blast=j.array([inj['bombBlast']],s.core.bomb_blast.dtype).reshape(1,13,15)))
                after,na,r,d,info,parts,sm=transition(s,aux,j.array([row['actions']]),jax.random.split(jax.random.PRNGKey(doc['seed']+tick),1),0.)
                jax.block_until_ready(r)
                if components is None:
                    components=np.zeros(parts.shape[-1]);component_names=list(E.ops.COMPONENTS[:-1])
                    component_names += [f'item_observed_{i}' for i in range(parts.shape[-1]-len(E.ops.COMPONENTS))]+['actual_total']
                fields=dict(carried=after.bun_carried[0],stored=after.bun_stored[0],loose=after.bun_loose[0].reshape(-1),alive=after.alive[0],respawn=after.bun_respawn[0],trapped=after.trapped[0])
                bad=[k for k,v in fields.items() if not np.array_equal(v,row['after'][k])]
                if np.max(np.abs(np.asarray(after.pos[0]).reshape(-1)-row['after']['pos']))>3e-4:bad.append('position')
                ph0=np.asarray(info['custody_phi_before'][0]);ph1=np.asarray(info['custody_phi_after'][0])
                if not np.allclose(ph0,row['phiBefore'],atol=1e-6) or not np.allclose(ph1,row['phiAfter'],atol=1e-6):bad.append('phi')
                if bad:mismatch.append(dict(tick=tick+1,fields=bad,js_phi_before=row['phiBefore'],js_phi_after=row['phiAfter'],jax_phi_before=ph0.tolist(),jax_phi_after=ph1.tolist(),
                    js_position=row['after']['pos'],jax_position=np.asarray(after.pos[0]).reshape(-1).tolist(),
                    js_carried=row['after']['carried'],jax_carried=np.asarray(after.bun_carried[0]).tolist()));break
                sums+=np.asarray(r[0]);custody+=np.asarray(info['custody_delta'][0]);terminal+=np.asarray(info['custody_terminal_pay'][0]);components+=np.asarray(parts[0,seat])
                if row['before']['stored'][seat][seat]>row['after']['stored'][seat][seat]:cycle_start=tick;cycle_raw=0.
                if cycle_start is not None:cycle_raw+=float(r[0,seat])
                if row['death'][1-seat]:interceptions.append(dict(tick=tick+1,kill=bool(info['surviving_kill'][0,seat]),native_popper=int(info['native_trap_popper'][0,1-seat]),delta=float(info['custody_delta'][0,seat])))
                if not row['before']['stored'][seat][seat] and row['after']['stored'][seat][seat]:
                    restores.append(dict(tick=tick+1,raw_reward=float(sums[seat]),raw_exact_cycle=cycle_raw,custody=float(custody[seat])))
                    cycle_start=None
                rowlog.append(dict(tick=tick+1,reward=np.asarray(r[0]).tolist(),custody=np.asarray(info['custody_delta'][0]).tolist(),terminal=np.asarray(info['custody_terminal_pay'][0]).tolist(),parts=np.asarray(parts[0,seat]).tolist()))
                s,aux=after,na
            tail=[]
            if mode in ('delivery','loss') and not mismatch:
                for tail_tick in range(12):
                    s,aux,rr,dd,ii,pp,sm=transition(s,aux,j.array([[[4,0],[4,0]]]),jax.random.split(jax.random.PRNGKey(9000+tail_tick),1),0.)
                    jax.block_until_ready(rr)
                    tail.append(dict(reward=np.asarray(rr[0]).tolist(),custody=np.asarray(ii['custody_delta'][0]).tolist(),terminal=np.asarray(ii['custody_terminal_pay'][0]).tolist()))
                if any(any(v!=0 for v in x['custody']+x['terminal']) for x in tail):failures.append(dict(seat=seat,mode=mode,reason='duplicate_terminal_or_custody_tail'))
            raw=OUT/f'P{seat}_{mode}_rewards.json';atomic_json(raw,rowlog)
            rec=dict(seat=seat,mode=mode,ticks=len(rowlog),expected_ticks=len(doc['rows']),fixture_sha256=sha(f),raw_rewards=str(raw),raw_sha256=sha(raw),
                real_composed_reward=sums.tolist(),custody_delta=custody.tolist(),terminal=terminal.tolist(),restores=restores,interceptions=interceptions,
                JAX_only_post_terminal_tail=tail,
                components={k:float(v) for k,v in zip(component_names,components)},mismatch=mismatch)
            records.append(rec)
            atomic_json(OUT/'RAW_RECORDS.json',records)
            if mismatch:failures.append(dict(seat=seat,mode=mode,reason='parity',details=mismatch));continue
            if mode in ('cycles','dropcycles'):
                if len(restores)!=5 or max(abs(x['custody']) for x in restores)>5e-6:failures.append(dict(seat=seat,reason='custody_cycle',restores=restores))
                if sums[seat]>5e-6:failures.append(dict(seat=seat,reason='full_composed_cycle_positive',raw=float(sums[seat]),components=rec['components']))
                if any(x['raw_exact_cycle']>1e-6 for x in restores):failures.append(dict(seat=seat,reason='isolated_cycle_positive',cycles=restores))
            if mode=='delivery' and not sums[seat]>0:failures.append(dict(seat=seat,reason='active_delivery_not_positive'))
            if mode=='loss' and not sums[seat]<0:failures.append(dict(seat=seat,reason='allowed_delivery_not_negative'))
            print(json.dumps({k:rec[k] for k in ('seat','mode','ticks','real_composed_reward','custody_delta','terminal','restores','mismatch')}),flush=True)
    # Same downstream legacy settlement as R1, ordinary far empty victim.
    names=set(L.TASK_EVENTS)|set(('death','mutual_death','threat_bomb_placed','safe_bomb_placed','tactical_bomb_enemy_safe_moves_before','tactical_bomb_enemy_safe_moves_after','danger_safe_resolution','surviving_kill','own_bomb_defeat','avoidable_danger_death','drop'))
    i={k:j.zeros((1,2),bool) for k in names};i['mutual_death']=j.zeros(1,bool)
    for k in ('tactical_bomb_enemy_safe_moves_before','tactical_bomb_enemy_safe_moves_after'):i[k]=j.zeros((1,2),j.int32)
    i.update(surviving_kill=j.array([[True,False]]),custody_kill_factor=j.ones((1,2)))
    _,a,_=L.retention_reward(i,L.reward_state(1),j.array([False]));ordinary=float(a['pending_kill'][0,0])
    close=float(R.graded_phi(False,True,False,0.,.2));far=float(R.graded_phi(False,True,False,0.,.8));loose=-.15
    monotonic=dict(q02_intercept=loose-close,q08_intercept=loose-far)
    if ordinary!=16. or not monotonic['q08_intercept']>monotonic['q02_intercept']>0:failures.append(dict(reason='ordinary_or_monotonic',ordinary=ordinary,monotonic=monotonic))
    result=dict(status='PASS' if not failures else 'BLOCKED',records=records,failures=failures,ordinary_kill_R1=16.,ordinary_kill_new=ordinary,monotonic=monotonic,
        source_sha256={f:sha(Path(f)) for f in ('scripts/bun_native_v2_custody_reward.py','scripts/bun_native_v2_custody_collector.py','scripts/bun_native_v2_custody_train.py')},
        limitations=['clear-brick map806; mirrored post-acquisition control/position intervention; not autonomous evaluation',
                     'Sim.step only, no human frameStep claim','conservative raw potential difference; no discounted policy-invariance claim'])
    atomic_json(OUT/'ACCEPTANCE.json',result)
    print(json.dumps(dict(status=result['status'],failures=failures,monotonic=monotonic)),flush=True)
    return result


if __name__=='__main__':run()
