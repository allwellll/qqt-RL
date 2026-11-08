"""Run in a fresh CPU process with eight virtual devices; no DCU is touched."""
import json
import os
# Keep the CPU mathematical reference independent of persistent ROCm caches.
os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'
import jax
import jax.numpy as j
import numpy as np
from qqt_rl.training.kill_ppo import build,gae


def forward(params,obs,state):
    logits=obs@params['actor']['w']
    vl=state@params['critic']['w']
    value=(jax.nn.softmax(vl)*j.linspace(-20,20,vl.shape[-1])).sum(-1)
    return logits[:,:5],logits[:,5:],value,vl


def main():
    devices=jax.local_devices();assert len(devices)==8
    rng=np.random.default_rng(78)
    params=dict(actor={'w':j.asarray(rng.normal(size=(2,8))*.03,j.float32)},
        critic={'w':j.asarray(rng.normal(size=(2,128))*.03,j.float32)})
    obs=rng.normal(size=(8,4,2)).astype('f4');state=rng.normal(size=(8,4,2)).astype('f4')
    batch=dict(obs=obs,state=state,mm=np.ones((8,4,5),bool),bm=np.ones((8,4,3),bool),
        action=np.zeros((8,4,2),np.int32),adv=rng.normal(size=(8,4)).astype('f4'),
        returns=rng.normal(size=(8,4)).astype('f4'),live=np.ones((8,4),'f4'))
    batch['live'][0]=0;batch['live'][1,:3]=0;batch['live'][4,:2]=0
    mv,ab,_,_=forward(params,j.asarray(obs.reshape(-1,2)),j.asarray(state.reshape(-1,2)))
    batch['logp']=np.asarray(jax.nn.log_softmax(mv)[:,0]+jax.nn.log_softmax(ab)[:,0]).reshape(8,4)
    init,_,multi=build(forward,devices,4)
    p=jax.device_put_replicated(params,devices);a=jax.device_put_replicated(init(params),devices)
    ref=jax.device_put_replicated(params['actor'],devices)
    keys=jax.random.split(jax.random.PRNGKey(1),8)
    (actual,actual_a),_=multi(p,a,ref,batch,keys,j.zeros(8,bool))
    # Same global batch on one device must yield the same aggregated optimizer
    # step, even with different numbers of live samples on individual shards.
    init1,_,single=build(forward,devices[:1],32)
    bp={k:np.reshape(v,(1,32)+v.shape[2:]) for k,v in batch.items()}
    (expected,expected_a),_=single(jax.device_put_replicated(params,devices[:1]),
        jax.device_put_replicated(init1(params),devices[:1]),
        jax.device_put_replicated(params['actor'],devices[:1]),bp,keys[:1],j.zeros(1,bool))
    for x,y in zip(jax.tree.leaves((actual,actual_a)),jax.tree.leaves((expected,expected_a))):
        for card in range(8):np.testing.assert_allclose(np.asarray(x[card]),np.asarray(y[0]),rtol=2e-4,atol=2e-7)
    (warm,warm_a),_=multi(p,a,ref,batch,keys,j.ones(8,bool))
    for x,y in zip(jax.tree.leaves(warm['actor']),jax.tree.leaves(p['actor'])):np.testing.assert_array_equal(x,y)
    for x,y in zip(jax.tree.leaves(warm_a['actor']),jax.tree.leaves(a['actor'])):np.testing.assert_array_equal(x,y)
    assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(warm['critic']),jax.tree.leaves(p['critic'])))
    # Terminal step must not bootstrap into the next independently reset game.
    adv,ret=gae(j.array([[1.],[2.]]),j.zeros((2,1)),j.array([[0.],[1.]]),j.array([100.]),gamma=1.,lam=1.)
    np.testing.assert_array_equal(ret,j.array([[3.],[2.]]))
    print(json.dumps(dict(status='PASS_CPU_8WAY_COLLECTIVES',devices=8,
        globally_weighted_gradient_matches_single_batch=True,replicas_equal=True,
        critic_warmup_freezes_actor_and_actor_adam=True,terminal_gae=True)))


if __name__=='__main__':main()
