"""Candidate process setup: one input-mask variable, no simulator mutation."""
from types import SimpleNamespace as NS
import jax.numpy as j
import numpy as np
import pytest


@pytest.mark.parametrize('common_encoder',[False,True])
def test_both_arms_share_encoder_and_only_treatment_rebinds_learner_mask(monkeypatch,common_encoder):
    from scripts import dagger_rl_r15_train as candidate
    original_obs=j.zeros((24,2,2))
    legacy_obs=lambda state,pid,danger=None,channels=24:original_obs[:channels]
    legacy_mask=lambda state:(j.ones((2,5),bool),j.ones((2,3),bool))
    physical_step=object()
    env=NS(make_obs=legacy_obs,legal_mask=legacy_mask,step=physical_step)
    outputs=[]
    for arm in ['legacy_mask','actual_mask']:
        train=NS(make_obs=legacy_obs,legal_mask=legacy_mask)
        monkeypatch.setattr(candidate.L.C,'E',NS(train=train,env=env))
        monkeypatch.setattr(candidate,'BASE_SETUP',lambda:('items','parent'))
        monkeypatch.setattr(candidate,'read',lambda path:dict(common_item_observation_alignment=common_encoder))
        monkeypatch.setattr(candidate,'ARM',arm)
        assert candidate.setup()==('items','parent')
        assert env.legal_mask is legacy_mask and env.step is physical_step
        assert (train.legal_mask is legacy_mask)==(arm=='legacy_mask')
        outputs.append(train.make_obs(NS(tactical_crate=j.ones((2,2))),0))
    np.testing.assert_array_equal(outputs[0],outputs[1])
    np.testing.assert_array_equal(outputs[0][7],np.full((2,2),int(common_encoder)))
