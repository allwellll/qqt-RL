"""Uninstalled candidate: scale only certified native-return PPO row terms.

No change to rewards, GAE, normalization, Critic, entropy, KL or replay. The
original Actor-row denominator and joint-action PPO clipping remain intact.
"""
import ast
import inspect
import textwrap

import jax
import jax.numpy as jnp
from scripts.native_return_retention import movement_scale

EVIDENCE_NAMES=('actor_rows','certified_rows','certified_positive_advantage_rows',
    'certified_disagreement_rows','certified_disagreement_positive_advantage_rows',
    'certified_disagreement_negative_route_positive_advantage_rows')


def replica_evidence(context,global_state,legal_move,legal_ability,advantage,actions):
    """Passive complete-rollout counts before optimizer ratio/clipping."""
    flatten=lambda x:x.reshape((-1,x.shape[-1]))
    c,g,m,b,a=map(flatten,(context,global_state,legal_move,legal_ability,actions))
    qualified=movement_scale(c,g,m,b)==0
    normalized=(advantage-advantage.mean())/(advantage.std()+1e-8)
    positive=normalized.reshape(-1)>0
    joint=c[:,95:110].reshape((-1,15))>0
    chosen=jnp.take_along_axis(joint,(a[:,0]*3+a[:,1])[:,None],axis=1)[:,0]
    wrong=qualified&~chosen
    return jnp.array([qualified.size,qualified.sum(),(qualified&positive).sum(),
        wrong.sum(),(wrong&positive).sum(),(wrong&positive&(c[:,81]<0)).sum()])


def patched_worker(source):
    initialization='    advantage_device_moments=jax.pmap(replica_moments)'
    iteration="        row['dagger']['learner_generation']=g-1"
    replacements=[(initialization,initialization+'''
    from scripts.retention_return_ppo import replica_evidence,EVIDENCE_NAMES
    return_ppo_device_evidence=jax.pmap(replica_evidence)'''),
        (iteration,iteration+'''
        ppo_evidence=np.asarray(return_ppo_device_evidence(batch[-1],batch[1],batch[6][0],
            batch[6][1],batch[4],batch[2])).sum(0)
        row['certified_return_ppo_evidence']=dict(zip(EVIDENCE_NAMES,map(int,ppo_evidence)))
        row['certified_return_ppo_evidence'].update(controls_training=False,
            scope='Complete rollout, exact certificate and replica PPO normalization; before ratio/clipping, not full gradient')''')]
    for anchor,replacement in replacements:
        assert source.count(anchor)==1
        source=source.replace(anchor,replacement)
    ast.parse(source)
    return source


def install_worker_telemetry(worker):
    current=worker.worker
    source=patched_worker(textwrap.dedent(inspect.getsource(current)))
    scope={}
    exec(compile(source,'<return-ppo-telemetry>','exec'),current.__globals__,scope)
    worker.worker=scope['worker']


def row_scale(context, global_state, legal_move, legal_ability, factor=1.):
    baseline=movement_scale(context,global_state,legal_move,legal_ability)
    return jax.lax.stop_gradient(factor+(1.-factor)*baseline)


def surrogate(logp, old_logp, advantage, context, global_state, legal_move,
              legal_ability, factor=1.):
    ratio=jnp.exp(logp-old_logp)
    clipped=jnp.maximum(-advantage*ratio,-advantage*jnp.clip(ratio,.9,1.1))
    return (clipped*row_scale(context,global_state,legal_move,legal_ability,factor)).mean()


def patched(source, factor):
    assert factor in [1.,.5], 'coefficient must be declared, never adapted by evaluation'
    anchor='        pol=jnp.maximum(-ad[ii[:half]]*ratio,-ad[ii[:half]]*jnp.clip(ratio,.9,1.1)).mean()'
    replacement=("        from scripts.retention_return_ppo import surrogate\n"
        "        pol=surrogate(logp,old[ii[:half]],ad[ii[:half]],\n"
        f"            context[ii[:half]],s[:half],m[:half],b[:half],{factor!r})")
    if replacement in source:
        assert anchor not in source
        return source
    assert source.count(anchor)==1, 'unexpected PPO call site'
    result=source.replace(anchor,replacement)
    ast.parse(result)
    return result


def install(factor):
    """Compile a function replacement in memory; never edit sealed source files."""
    from scripts import je09_stage_h5c_ops as ops
    from scripts import je09_stage_h6_ops, je09_stage_h8_ops, je10_ops
    aliases=[ops,je09_stage_h6_ops,je09_stage_h8_ops,je10_ops]
    current=ops.ppo
    if getattr(current,'_return_ppo_factor',None) is not None:
        assert current._return_ppo_factor==factor
        assert all(module.ppo is current for module in aliases)
        return current
    assert all(module.ppo is current for module in aliases),'unexpected worker PPO override'
    source=patched(textwrap.dedent(inspect.getsource(current)),factor)
    scope={}
    exec(compile(source,'<certified-return-ppo>','exec'),current.__globals__,scope)
    installed=scope['ppo']
    installed._return_ppo_factor=factor
    for module in aliases:
        module.ppo=installed
    return installed
