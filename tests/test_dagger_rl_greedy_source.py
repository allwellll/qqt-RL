import jax.numpy as j
import numpy as np

from scripts.dagger_rl_greedy_source import auxiliary_action


def test_sampled_control_preserves_actions_and_log_probability_exactly():
    move=j.array([[10.,0.,0.,0.,0.]])
    ability=j.array([[9.,0.,0.]])
    own=j.array([[3,2]]);old=j.array([-5.])
    actual,prob=auxiliary_action(move,ability,j.ones((1,5),bool),j.ones((1,3),bool),own,old,greedy=False)
    np.testing.assert_array_equal(actual,own);np.testing.assert_array_equal(prob,old)


def test_auxiliary_greedy_honors_masks_and_has_actual_action_likelihood():
    move=j.array([[10.,2.,0.,0.,0.]])
    ability=j.array([[1.,0.,8.]])
    own,lp=auxiliary_action(move,ability,j.array([[False,True,True,True,True]]),
                           j.array([[True,True,False]]),j.array([[3,2]]),j.array([-5.]),greedy=True)
    np.testing.assert_array_equal(own,[[1,0]])
    assert np.isfinite(np.asarray(lp)).all() and float(lp[0])<0
