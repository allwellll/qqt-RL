import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import jax
import jax.numpy as j
import numpy as np
import optax
from jax_bomb import jax_net as net
from scripts import r3_items_observation as O


def test_three_fixed_states_logits_argmax_and_zero_only_expansion():
    p=net.init_transformer(jax.random.PRNGKey(120),24,13,15,embed=8,depth=1,heads=4)
    # Nonzero novel inputs: equality must follow zero new weights, not absent
    # exposure. Combat/window/carry span all three required fixed states.
    obs=jax.random.uniform(jax.random.PRNGKey(121),(3,24,13,15))
    gv=j.zeros((3,24)).at[:,9:11].set(1).at[1,10].set(1.5).at[2,13:15].set(1)
    expanded_obs=j.concatenate([obs,j.ones((3,1,13,15))*.375],1)
    expanded_gv=j.concatenate([gv,j.ones((3,13))*.7],-1)
    old=O.phase.forward(p,obs,gv);new=O.forward(O.migrate(p),expanded_obs,expanded_gv)
    for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
    for a,b in zip(old[:2],new[:2]):np.testing.assert_array_equal(j.argmax(a,-1),j.argmax(b,-1))
    q=O.migrate(p);np.testing.assert_array_equal(q['tok'][0][:216],p['tok'][0]);np.testing.assert_array_equal(q['tok'][0][216:],0)
    np.testing.assert_array_equal(q['state_w'][:24],p['state_w']);np.testing.assert_array_equal(q['state_w'][24:],0)


def test_optimizer_preserves_count_and_old_moments():
    p=net.init_transformer(jax.random.PRNGKey(123),24,13,15,embed=8,depth=1,heads=4)
    opt=optax.adam(.000075);adam=opt.init(p)
    _,adam=opt.update(jax.tree.map(j.ones_like,p),adam,p)
    new=O.migrate(adam)
    assert int(new[0].count)==int(adam[0].count)==1
    np.testing.assert_array_equal(new[0].mu['state_w'][:24],adam[0].mu['state_w'])
    np.testing.assert_array_equal(new[0].mu['state_w'][24:],0)


def test_new_input_rows_receive_real_actor_gradient():
    p=O.migrate(net.init_transformer(jax.random.PRNGKey(125),24,13,15,embed=8,depth=1,heads=4))
    obs=j.ones((1,25,13,15))*.2;gv=j.ones((1,37))*.3
    grad=jax.grad(lambda w:O.forward(dict(p,state_w=w),obs,gv)[1][0,2])(p['state_w'])
    assert np.linalg.norm(np.asarray(grad[24:]))>0
