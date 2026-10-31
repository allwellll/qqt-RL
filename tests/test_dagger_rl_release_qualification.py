"""Joint route labels must distinguish legal release from certified utility."""
import jax
import jax.numpy as j
import numpy as np

from scripts.dagger_rl_oracle import combine, set_loss
from scripts.dagger_rl_release_qualification import qualified_teacher


def fixture():
    route=j.zeros((4,5,3),bool).at[:,2,:].set(True)
    item=j.zeros_like(route).at[1,1,2].set(True).at[3,0,0].set(True)
    banana=j.array([True,True,False,True])
    return route,item,banana


def test_control_exact_and_inputs_unchanged():
    route,item,banana=fixture()
    label,stats=qualified_teacher(route,item,banana,enabled=False)
    np.testing.assert_array_equal(label,combine(route,item))
    assert route[0,2,2] and item[1,1,2]
    assert int(stats[1])==0


def test_only_uncertified_banana_release_removed_no_move_or_item_changes():
    route,item,banana=fixture()
    label,stats=jax.jit(qualified_teacher,static_argnames='enabled')(route,item,banana,enabled=True)
    expected=combine(route,item).at[0,2,2].set(False)
    np.testing.assert_array_equal(label,expected)
    assert int(stats[0])==1 and int(stats[1])==1
    # Every item-oracle set, including owned contact with ability zero, survives.
    np.testing.assert_array_equal(label[1],item[1])
    np.testing.assert_array_equal(label[3],item[3])


def test_actual_set_loss_ability_gradient_no_longer_cancels():
    route,item,banana=fixture()
    move=j.zeros((1,5));ability=j.array([[.1,-.2,.3]])
    old=combine(route,item)[:1]
    new,_=qualified_teacher(route,item,banana,enabled=True)
    derivative=lambda mask:jax.grad(lambda a:set_loss(move,a,mask))(ability)
    assert abs(float(derivative(old)[0,2]))<1e-6
    assert float(derivative(new[:1])[0,2])>.1
    assert float(derivative(item[1:2])[0,2])<-.1


def test_abstains_instead_of_inventing_legal_actions_when_set_empties():
    route=j.zeros((1,5,3),bool).at[0,0,2].set(True)
    item=j.zeros_like(route)
    label,stats=qualified_teacher(route,item,j.array([True]),enabled=True)
    assert not label.any() and int(stats[2])==1
    assert float(set_loss(j.zeros((1,5)),j.zeros((1,3)),label))==0


def test_combat_empty_route_and_glue_unchanged():
    route=j.zeros((2,5,3),bool).at[1,3,:].set(True)
    item=j.zeros_like(route)
    label,_=qualified_teacher(route,item,j.array([True,False]),enabled=True)
    np.testing.assert_array_equal(label,route)
