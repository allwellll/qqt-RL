import jax.numpy as jnp
import numpy as np
from scripts.return_detour_replay import detour_flags


def sample(moves, *, position=(6, 10), valid=True, kind=3, preparing=False,
           carry=True):
    n=len(moves)
    obs=jnp.zeros((n,24,13,15)).at[:,0,position[0],position[1]].set(1.)
    obs=obs.at[:,14,1:4,4:7].set(1.)
    joint=jnp.zeros((n,5,3),bool)
    for i,accepted in enumerate(moves):
        joint=joint.at[i,jnp.array(accepted),0].set(True)
    return dict(obs=obs, gv=jnp.zeros((n,24)).at[:,13].set(float(carry)),
        joint_target=joint, legal=jnp.ones((n,5),bool),
        ability_legal=jnp.ones((n,3),bool),valid=jnp.full(n,valid),
        kind=jnp.full(n,kind),preparation=jnp.full(n,preparing))


def test_only_sets_whose_every_accepted_move_goes_away_are_detours():
    data=sample([[3],[1],[1,3],[2],[0],[2,3],[4],[3,4]])
    np.testing.assert_array_equal(detour_flags(data),
        [True,True,True,False,False,False,False,False])


def test_native_legal_joint_set_controls_classification():
    data=sample([[2,3],[2,3],[2,3]])
    data['legal']=data['legal'].at[0,2].set(False)
    data['joint_target']=data['joint_target'].at[1,3,0].set(False)
    data['joint_target']=data['joint_target'].at[2,2,0].set(False).at[2,2,1].set(True)
    data['ability_legal']=data['ability_legal'].at[2,1].set(False)
    np.testing.assert_array_equal(detour_flags(data),[True,False,True])


def test_detour_class_never_imports_other_phases_or_empty_custody():
    for kwargs in [dict(valid=False),dict(kind=2),dict(preparing=True),dict(carry=False)]:
        assert not bool(detour_flags(sample([[3]],**kwargs))[0])


def test_cage_delivery_and_missing_cage_are_not_detour_labels():
    data=sample([[3]],position=(2,5))
    assert not bool(detour_flags(data)[0])
    data=sample([[3]]);data['obs']=data['obs'].at[:,14].set(0.)
    assert not bool(detour_flags(data)[0])
