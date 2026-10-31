"""CPU proof: passive counters and equal fixed actors preserve real actions/rewards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts.population_run_state import atomic_json, sha


def main():
    started=time.time()
    from scripts.r3_items_runtime import install
    install()
    from scripts.bun_native_v2_custody_fast2 import install as fast
    fast()
    from scripts.bun_spawn_side import install as ownside
    ownside()
    from scripts import ppo_scale_longrun_train as P, r3_items_collector as M
    P.T.configure();P.C.install_provenance();M.configure(True,True)
    keys=jax.random.split(jax.random.PRNGKey(2026323100),18)
    snapshot=Path('runs/r3_items_route_20261030/formal/r3_a_c000000/u003040/state.pkl')
    payload=pickle.load(snapshot.open('rb'))
    st,ax=jax.tree.map(lambda value:j.asarray(value[0,:18]),(payload['states'],payload['aux']))
    del payload
    kinds=j.arange(18)//3;seats=j.arange(18)%2;scene=j.zeros(18,bool)
    ax=dict(ax,combat_mix=M.metadata(st,kinds,scene,seats))
    pool=Path('runs/r3_items_route_20261030/formal/r3_a_c000000/u003040/policy.pt')
    actor=pickle.load(pool.open('rb'));params=dict(actor=actor,critic=actor)
    groups={name:np.flatnonzero(np.asarray(kinds)==i) for i,name in [(3,'current'),(4,'ring'),(5,'fixed')]}
    groups['bot']=np.flatnonzero(np.asarray(kinds)<3)
    old_choose=M.choose;old_transition=M.transition
    old=jax.jit(lambda st,ax,keys:old_choose(params,actor,actor,st,ax,keys,groups))(st,ax,keys)
    jax.block_until_ready(old)
    from scripts.r3_selfplay_monitor import install as monitor
    monitor(M)
    monitored_ax=dict(ax,combat_mix=M.metadata(st,kinds,scene,seats))
    newgroups=dict(groups)
    for identity in range(3):newgroups['fixed'+str(identity)]=groups['fixed'][groups['fixed']%3==identity]
    new=jax.jit(lambda st,ax,keys:M.choose(params,actor,(actor,actor,actor),st,ax,keys,newgroups))(st,monitored_ax,keys)
    jax.block_until_ready(new)
    assert jax.tree.structure(old)==jax.tree.structure(new)
    for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(new)):np.testing.assert_array_equal(a,b)
    actions=old[2];stepkeys=old[1]
    original=jax.jit(lambda st,ax:old_transition(st,ax,actions,stepkeys,0.))(st,ax)
    observed=jax.jit(lambda st,ax:M.transition(st,ax,actions,stepkeys,0.))(st,monitored_ax)
    jax.block_until_ready(observed)
    # Remove only the newly declared passive ledger/metrics before comparison.
    after,aux,reward,done,info,parts,metrics=observed
    passive={k:v for k,v in metrics.items() if k.startswith('r3_sp_')}
    assert all(np.asarray(v).shape==(18,3,2) for v in passive.values())
    assert all(np.asarray(v).sum(0).shape==(3,2) for v in passive.values())
    aux=dict(aux,combat_mix={k:v for k,v in aux['combat_mix'].items() if k!='sp_monitor'})
    metrics={k:v for k,v in metrics.items() if not k.startswith('r3_sp_')}
    comparable=(after,aux,reward,done,info,parts,metrics)
    assert jax.tree.structure(original)==jax.tree.structure(comparable)
    for a,b in zip(jax.tree.leaves(original),jax.tree.leaves(comparable)):np.testing.assert_array_equal(a,b)
    atomic_json(Path('runs/r3_selfplay_ramp_20261031/CPU_RUNTIME_ACCEPTANCE.json'),dict(status='PASS',
        real_native_states=18,all_six_opponent_kinds=True,both_physical_seats=True,all_three_fixed_subgroups_exercised=True,equal_fixed_actor_actions_bitwise=True,
        original_state_reward_done_info_parts_metrics_exact=True,only_passive_aux_and_metrics_added=True,
        collector_metric_ABI='environment,opponent3,seat2 -> one collector sum -> opponent3,seat2',
        actor_sha256=sha(pool),source_sha256={f:sha(Path(f)) for f in ['scripts/r3_selfplay_monitor.py','scripts/r3_selfplay_train_20261031.py']},
        wall_seconds=time.time()-started,unix=time.time()))
    print('R3-sp passive action/reward parity PASS',time.time()-started)


if __name__=='__main__':main()
