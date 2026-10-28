"""Joint replay uses the real split Actor/Critic forward and Actor gradients."""
import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import jax
import jax.numpy as jnp
import numpy as np


def test_joint_replay_updates_real_actor_move_and_ability_while_critic_stays_zero(monkeypatch):
    from jax_bomb import jax_train as train
    from scripts.je03_bun_stage1_20261011 import split_forward
    from scripts.prestock_joint_replay import replay_loss
    actor=train.init_net(jax.random.PRNGKey(4),'transformer',24,13,15,embed=16,depth=1)
    critic=train.init_net(jax.random.PRNGKey(5),'transformer',24,13,15,embed=16,depth=1)
    params=dict(actor=actor,critic=critic)
    original=train.net_forward
    monkeypatch.setattr(train,'net_forward',lambda p,a,o,s=None:split_forward(p,a,o,s,original))
    replay=dict(obs=jax.random.normal(jax.random.PRNGKey(8),(2,24,13,15)),gv=jnp.ones((2,24)),
        legal=jnp.ones((2,5),bool),ability_legal=jnp.ones((2,3),bool),
        joint_target=jnp.zeros((2,5,3),bool).at[:,2,1].set(True),valid=jnp.ones(2,bool),
        kind=jnp.full(2,3,jnp.int32),preparation=jnp.zeros(2,bool))
    value,grad=jax.jit(jax.value_and_grad(replay_loss))(params,replay)
    assert np.isfinite(float(value)) and float(value)>0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    norm=lambda tree:float(jnp.sqrt(sum((x*x).sum() for x in jax.tree.leaves(tree))))
    assert norm(grad['actor']['heads']['wm'])>0 and norm(grad['actor']['heads']['wb'])>0
    assert norm(grad['actor']['tok'])>0 and norm(grad['critic'])==0
    # A real parameter update should increase this indivisible pair's mass.
    updated=jax.tree.map(lambda p,g:p-.01*g,params,grad)
    assert float(replay_loss(updated,replay))<float(value)
