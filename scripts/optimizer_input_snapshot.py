"""Passive first-PPO-minibatch evidence; never modifies optimizer inputs."""
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def first_microbatch(batch,keys,minibatch):
    """Select original Actor/Critic permutations before any optimizer step."""
    shape=batch[4].shape
    if len(shape)!=3 or minibatch<=0 or minibatch%2 or np.prod(shape[1:])%(minibatch//2):
        raise ValueError('invalid first microbatch shape')
    total=int(np.prod(shape[1:]));half=minibatch//2
    if half>total:raise ValueError('minibatch exceeds complete rollout')
    def one(data,key):
        ka,kc=jax.random.split(key)
        ids=jnp.concatenate([jax.random.permutation(ka,total)[:half],jax.random.permutation(kc,total)[:half]])
        flatten=lambda x:x.reshape((total,)+x.shape[2:])
        compact=jax.tree.map(lambda x:flatten(x)[ids],data)
        adv=data[4];norm=(adv-adv.mean())/(adv.std()+1e-8)
        return ids,compact,norm.reshape(-1)[ids]
    indices,compact,norm=jax.jit(jax.vmap(one))(batch,keys)
    return dict(indices=indices,batch=compact,normalized_advantage=norm,actor_rows_per_device=half)


def _digest(tree):
    h=hashlib.sha256();leaves,structure=jax.tree.flatten(tree);h.update(str(structure).encode())
    for x in leaves:
        x=np.asarray(x);h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.tobytes())
    return h.hexdigest()


def capture(base,generation,params,adam,batch,keys,replay,minibatch):
    base=Path(base);directory=base/'diagnostics';directory.mkdir(parents=True,exist_ok=True)
    payload_name=f'diagnostics/OPTIMIZER_INPUT_{generation:06d}.pkl';path=base/payload_name
    marker=directory/f'OPTIMIZER_INPUT_{generation:06d}.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=saved['sha256']:raise ValueError('snapshot digest mismatch')
        return saved
    devices=batch[4].shape[0]
    def unrep(tree):
        def one(x):
            x=np.asarray(x)
            if x.shape[0]!=devices or not all(np.array_equal(x[0],r) for r in x[1:]):raise ValueError('replica disagreement')
            return x[0]
        return jax.tree.map(one,tree)
    raw_params=unrep(params);raw_adam=unrep(adam)
    selected=first_microbatch(batch,keys,minibatch)
    saved=dict(schema='actual-first-ppo-input-v1',generation=generation,params=raw_params,adam=raw_adam,
        keys=np.asarray(keys),full_raw_advantages=np.asarray(batch[4]),
        selection=jax.device_get(selected),dagger_replay=jax.device_get(replay),
        source_actor_digest=_digest(raw_params['actor']),source_critic_digest=_digest(raw_params['critic']) if 'critic' in raw_params else None)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:pickle.dump(saved,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    receipt=dict(passed=True,schema=saved['schema'],generation=generation,payload=payload_name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_actor_digest=saved['source_actor_digest'],
        source_critic_digest=saved['source_critic_digest'],devices=devices,minibatch_per_device=minibatch,
        actor_rows_per_device=minibatch//2,full_rollout_rows_per_device=int(np.prod(batch[4].shape[1:])),
        original_full_rollout_advantage_saved=True,optimizer_steps_added=0,new_games=0,controls_training=False,
        scope='Pre-update raw parameters/Adam, first actual Actor and Critic permutation rows and native replay. No subset GAE renormalization, sampling/RNG/state/optimizer mutation, or training import. Selection uses original production formula; independently compiled normalization can have device fusion precision differences. Snapshot success never controls training.')
    temp_marker=marker.with_suffix('.tmp');temp_marker.write_text(json.dumps(receipt,indent=2)+'\n');temp_marker.replace(marker)
    return receipt
