"""Isolated ordinary native kill reference/revision comparison, CPU only."""
import json,sys,subprocess
from pathlib import Path
import jax,jax.numpy as j,numpy as np
from scripts.population_run_state import atomic_json,sha


def run():
    from scripts.bun_native_v2_runtime import install
    install()
    from scripts import carrier_mechanism_engine as E
    E.configure()
    version=sys.argv[1]
    if version=='new':
        from scripts.bun_native_v2_custody_reward import install as reward_install
        reward_install()
    else:
        from scripts.bun_native_v2_reward import install as reward_install
        reward_install(False)
    from scripts.bun_native_v2_parity import fresh
    out=Path('runs/bun_native_v2_20261030/custody_v2_tests');results=[]
    fn=jax.jit(E.transition)
    for seat in (0,1):
        f=out/f'P{seat}_ordinary.json'
        if not f.exists():f.write_bytes(subprocess.check_output(['node','scripts/bun_native_v2_custody_fixture.js',str(seat),'ordinary']))
        doc=json.loads(f.read_text());single=fresh(doc);d=doc['initial']
        single=single._replace(core=single.core._replace(fuse=j.array(d['fuse'],single.fuse.dtype).reshape(13,15),
            owner=j.array(d['owner'],single.core.owner.dtype).reshape(13,15),bomb_blast=j.array(d['bombBlast'],single.core.bomb_blast.dtype).reshape(13,15)))
        st=jax.tree.map(lambda x:x[None],single);aux=E.ops.reward_state(1);parts_sum=np.zeros(len(E.ops.COMPONENTS));kills=[];errors=[]
        for tick,row in enumerate(doc['rows']):
            inj=row['injection']
            if inj:st=st._replace(core=st.core._replace(pos=j.array([inj['pos']],j.float32).reshape(1,2,2),blast_linger=j.array([inj['blastLinger']],st.core.blast_linger.dtype).reshape(1,13,15)))
            after,na,r,done,info,parts=fn(st,aux,j.array([row['actions']]),jax.random.split(jax.random.PRNGKey(doc['seed']+tick),1))
            jax.block_until_ready(r)
            if not np.array_equal(after.alive[0],row['after']['alive']) or not np.array_equal(after.trapped[0],row['after']['trapped']):errors.append(tick+1)
            parts_sum+=np.asarray(parts[0,seat])
            if info['death'][0,1-seat]:kills.append(dict(tick=tick+1,causal=bool(info['causal_kill'][0,seat]),popper=int(info['native_trap_popper'][0,1-seat])))
            st,aux=after,na
        names=['settled_kill','control_entry_reward','control_residual_reward']
        results.append(dict(seat=seat,ordinary_kill_components={k:float(parts_sum[E.ops.COMPONENTS.index(k)]) for k in names},
            kills=kills,parity_failures=errors,fixture_sha256=sha(f),ticks=len(doc['rows'])))
    result=dict(status='PASS' if all(not r['parity_failures'] and r['kills'] for r in results) else 'BLOCKED',version=version,records=results,
        limitations=['preplaced causal bomb fixed fixture; real explosion/trap/contact death in both engines','no autonomous performance claim'])
    atomic_json(out/f'ORDINARY_{version}.json',result);print(json.dumps(result))


if __name__=='__main__':run()
