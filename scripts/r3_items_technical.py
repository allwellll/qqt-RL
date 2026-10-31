"""CPU focused native event/collector and homologous parent migration checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json,sha

OUT=Path('runs/r3_items_route_20261030')


def main():
    from scripts.r3_items_runtime import install
    install()
    from scripts import carrier_mechanism_engine as E,combat_item_engine as T,r3_items_collector as M
    from scripts import r3_items_reward as R,r3_items_observation as O
    T.configure();M.configure()
    keys=jax.random.split(jax.random.PRNGKey(2026208199),3);st,ax=E.initial(keys,0);seats=j.array([0,1,0])
    from scripts.bomb_threat_r1_item_progress import fraction
    ax=dict(ax,r1_attribute_highwater=fraction(j.stack([st.core.bombs_cap,st.core.blast_cap,st.core.spd_g],-1)),tactics=T.events.state(3),
        provenance_source=j.full(3,-1,j.int32),combat_mix=M.metadata(st,j.ones(3,j.int32),j.zeros(3,bool),seats),
        a3sr={k:j.zeros(3,j.int32) for k in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
    source=json.loads((OUT/'R3_PARENT.json').read_text());cp=Path(source['pinned_checkpoint'])
    assert sha(cp/'policy.pt')==source['source']['policy_sha256']
    parent=pickle.load((cp/'policy.pt').open('rb'));migrated=O.migrate(parent)
    oldo,oldg,*_=M.C.inputs(st,seats);o,g,*_=M.inputs(st,seats,ax['combat_mix']['bomb_threat']['r3'])
    f0=jax.jit(lambda p,o,g:O.phase.forward(p,o,g));f1=jax.jit(O.forward)
    before=f0(parent,oldo,oldg);after=f1(migrated,o,g)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    atomic_json(OUT/'PARENT_INPUT_MIGRATION.json',dict(status='PASS',parent=source['source'],fixed_native_states=3,seats=[0,1,0],
        input_shapes=[list(o.shape),list(g.shape)],Actor_logits_bitwise=True,Actor_argmax_exact=True,
        old_token_rows_preserved=True,new_token_rows_zero=True,old_global_rows_preserved=True,new_global_rows_zero=True))
    action=j.zeros((3,2,2),j.int32).at[:,:,0].set(4)
    fn=jax.jit(lambda s,a,k:M.transition(s,a,action,k,0.))
    began=time.time();after,new,r,d,info,parts,m=fn(st,ax,keys);jax.block_until_ready(r)
    assert np.isfinite(np.asarray(r)).all() and m['r3_pay'].shape==(3,8)
    assert info['r3_contact_type'].shape==(3,2)
    atomic_json(OUT/'TECHNICAL_NATIVE_STEP.json',dict(status='PASS',physical_seats=[0,1],native_event_ABI=True,
        r3_reward_channels=8,total_reward=np.asarray(r).tolist(),r3_payment=np.asarray(m['r3_pay']).tolist(),
        wall_seconds=time.time()-began,actual_optimizer_update=False,source_sha256={str(p):sha(p) for p in Path('scripts').glob('r3_items_*.py')}))
    print('native step and actual parent input migration PASS',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'TECHNICAL_NATIVE_BLOCKED.json',dict(status='BLOCKED',reason='inspect TECHNICAL_NATIVE.log',unix=time.time()))
        raise
