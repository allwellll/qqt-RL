"""One real small on-policy PPO update with frozen versus passive collector.

CPU-only technical evidence; no ability inference and no formal budget reuse.
"""
import copy
import json
import time
from pathlib import Path


def main():
    import jax
    import jax.numpy as j
    import numpy as np
    import optax
    from scripts import ppo_scale_longrun_train as P, bomb_threat_collector as old
    from scripts import bomb_threat_collector_diagnostics_v1 as new
    from scripts import bomb_threat_iter2_recipe as recipe
    from scripts.bomb_threat_runtime import install
    from scripts.carrier_mechanism_analysis import bank, merged
    from scripts.checkpoint_policy_artifact import load_actor
    from scripts.terminal_checkpoint_restore import digest
    from scripts.population_run_state import atomic_json, sha
    root=Path(__file__).resolve().parents[1];base=root/'runs/bomb_threat_v1'
    config=json.loads((base/'iter2/TRAINING_PROTOCOL.json').read_text())
    P.T.configure();P.C.install_provenance();install()
    actor=load_actor(root/config['policy_source']['checkpoint'])
    params=dict(actor=actor,critic=copy.deepcopy(actor));opt=optax.adam(7.5e-5,eps=1e-8)
    adam={k:opt.init(v) for k,v in params.items()}
    rep=lambda tree:jax.tree.map(lambda a:j.asarray(a)[None],tree)
    items=[x for x in bank() if x['record']['split']=='train']
    bs,ba=merged(items,'states'),merged(items,'aux');n=2;seed=2026112920
    keys=jax.random.split(jax.random.PRNGKey(seed),n)
    spawns=json.loads((root/config['spawn_manifest']).read_text())['spawn_cells']
    weights=[0.,0.,0.,1.,0.,0.];designated=j.array([False,False])
    st,ax=jax.jit(old.arena_fresh)(keys,bs,ba,designated,j.asarray(spawns))
    ax=dict(ax,combat_mix=old.metadata(st,j.array([3,3]),designated))
    args=(rep(params),rep(st),rep(ax),rep(keys),bs,ba,rep(actor),rep(actor))
    start=time.time()
    baseline=old.collector(2,n,seed,.0,0.,spawns,weights)(*args)
    measured=new.collector(2,n,seed,.0,0.,spawns,weights)(*args)
    jax.block_until_ready(measured)
    def exact(a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)
    exact(baseline[:2],measured[:2])
    for k,v in baseline[2].items():exact(v,measured[2][k])
    update=recipe.updater(2,opt,lambda p,b,e:P.pure_ppo_objective(p,b))
    random=jax.random.split(jax.random.PRNGKey(seed+1),1)
    u0=update(rep(params),rep(adam),baseline[1],random,.005,1.,False)
    u1=update(rep(params),rep(adam),measured[1],random,.005,1.,False)
    jax.block_until_ready(u1);exact(u0,u1)
    l0=np.asarray(u0[1]).mean((0,1));l1=np.asarray(u1[1]).mean((0,1))
    delta0=P.norms(params,P.one(u0[0][0]));delta1=P.norms(params,P.one(u1[0][0]))
    assert all(v>0 and np.isfinite(v) for v in delta0.values())
    diagnostic=jax.tree.map(lambda a:np.asarray(a).sum(0).tolist(),jax.device_get(measured[2]))
    evidence=dict(status='PASS',platform=jax.default_backend(),environments=n,rollout_ticks=2,seed=seed,
        true_on_policy_steps=4,Adam_steps=2,production_actor=config['policy_source'],
        exact_actions_env_aux_RNG=True,exact_PPO_batch=True,exact_params_Adam_loss=True,
        before=dict(loss=l0.tolist(),update_l2=delta0),after=dict(loss=l1.tolist(),update_l2=delta1),
        params_Adam_sha256=digest(u0[0]),diagnostic={k:diagnostic[k] for k in ('policy_entropy_sum','bomb_probability_sum','bomb_argmax_count','policy_diagnostic_ticks')},
        source_sha256={f:sha(root/f) for f in ('scripts/bomb_threat_collector.py','scripts/bomb_threat_collector_diagnostics_v1.py','scripts/bomb_threat_passive_diagnostics.py')},
        elapsed_seconds=time.time()-start,scope='one identical CPU controlled update; not equality of adjacent stochastic live updates',unix=time.time())
    atomic_json(base/'iter2/STEER1120_UPDATE_PARITY.json',evidence)
    print(json.dumps(evidence))


if __name__=='__main__':main()
