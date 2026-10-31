import json
import numpy as np
import pytest
from scripts import rules_v2_critic_probes as p


def test_new_rule_probes_refuse_old_rule_or_blocked_gates(tmp_path):
    (tmp_path/'GATES.json').write_text(json.dumps(dict(G1={'status':'BLOCKED'},G2={'status':'NOT_STARTED'})))
    with pytest.raises(RuntimeError,match='G1 blocked'):p.require_rules_ready(tmp_path)


def test_terminal_mc_does_not_use_censored_tail_or_tested_value():
    rewards=np.array([[[1.,2.,0.],[2.,3.,4.]]])
    valid=np.array([[[1,1,0],[1,1,1]]],bool)
    terminal=np.array([[[0,1,0],[0,0,1]]],bool)
    np.testing.assert_allclose(p.terminal_mc_returns(rewards,valid,terminal),[[1+.995*2,2+.995*3+.995**2*4]])
    terminal[0,1,2]=False
    with pytest.raises(ValueError,match='Censored'):p.terminal_mc_returns(rewards,valid,terminal)


def test_noncontiguous_mc_and_fake_early_terminal_are_rejected():
    r=np.ones((1,1,3));v=np.array([[[1,0,1]]],bool);t=np.array([[[0,0,1]]],bool)
    with pytest.raises(ValueError,match='Noncontiguous'):p.terminal_mc_returns(r,v,t)
    with pytest.raises(ValueError,match='invalid terminal'):p.terminal_mc_returns(r,np.ones_like(v),np.ones_like(t))
