from types import SimpleNamespace as NS
import jax.numpy as j
import numpy as np
from scripts.dagger_rl_movement_probe import actual_speed_mask


def fixture(*,trapped=0,lesson=0,carried=1):
    # Pure mask-function unit inputs, never environment initialization/training.
    old=j.array([[False,True,True,True,True],[True,False,True,True,True]])
    changed=j.ones((2,5),bool);ability=j.array([[True,False,True],[True,True,False]])
    core=NS(spd_g=j.array([2.,1.]))
    core._replace=lambda **kw:NS(**kw)
    state=NS(core=core,alive=j.array([True,True]),trapped=j.array([trapped,0]),lesson=j.array(lesson),bun_carried=j.array([carried,-1]))
    calls=[]
    def probe(c):calls.append(c.spd_g);return changed,j.zeros((2,2),bool)
    env=NS(base=NS(legal_mask=probe),_movement_scale=lambda s:j.array([.3,1.]),LESSON_FULL=0)
    return state,lambda s:(old,ability),env,old,changed,ability,calls


def test_enabled_uses_physics_speed_only_for_live_full_lesson_learner_movement():
    st,legacy,env,old,new,ability,calls=fixture()
    mm,bm=actual_speed_mask(legacy,env,enabled=True)(st)
    np.testing.assert_array_equal(mm[0],new[0]);np.testing.assert_array_equal(mm[1],old[1])
    assert bm is ability
    np.testing.assert_allclose(calls[0],[.6,1.])


def test_disabled_is_exact_legacy_without_extra_probe():
    st,legacy,env,old,new,ability,calls=fixture()
    mm,bm=actual_speed_mask(legacy,env,enabled=False)(st)
    assert mm is old and bm is ability and not calls


def test_inactive_or_scripted_lessons_keep_original_mask():
    for trapped,lesson in [(1,0),(0,1)]:
        st,legacy,env,old,new,ability,calls=fixture(trapped=trapped,lesson=lesson)
        mm,bm=actual_speed_mask(legacy,env,enabled=True)(st)
        np.testing.assert_array_equal(mm,old);assert bm is ability


def test_empty_combat_remains_exact_even_with_slow_status():
    st,legacy,env,old,new,ability,calls=fixture(carried=-1)
    mm,bm=actual_speed_mask(legacy,env,enabled=True)(st)
    np.testing.assert_array_equal(mm,old);assert bm is ability


def test_training_install_only_changes_collector_binding_and_disabled_is_identity():
    from scripts.dagger_rl_movement_probe import install_learner_mask
    st,legacy,env,old,new,ability,calls=fixture()
    env.legal_mask=legacy;step=object();env.step=step
    train=NS(legal_mask=legacy)
    assert install_learner_mask(train,env,enabled=False) is legacy
    assert train.legal_mask is legacy
    installed=install_learner_mask(train,env,enabled=True)
    assert train.legal_mask is installed and installed is not legacy
    assert env.legal_mask is legacy and env.step is step
    np.testing.assert_array_equal(installed(st)[0][0],new[0])
