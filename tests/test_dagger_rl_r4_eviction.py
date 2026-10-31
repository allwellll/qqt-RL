"""Eviction is the only candidate change; both arms keep the screened anchor."""
import ast
from pathlib import Path
import numpy as np
import pytest


def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}


def test_core_math_unchanged_and_both_arms_use_same_parent_anchor():
    prior=functions('scripts/dagger_rl_r3_train.py');new=functions('scripts/dagger_rl_r4_train.py')
    for name in ['install_provenance','initial','collector','supervised','objective','updater','commit']:
        assert prior[name]==new[name],name
    code=Path('scripts/dagger_rl_r4_train.py').read_text()
    assert "mode='anchor' if arm" not in code
    assert "mode='anchor'" in code
    assert 'def next_update_probe(' in code, 'fresh restore must verify an executed on-policy collector+memory+optimizer continuation'


def test_fifo_control_has_exact_prior_training_replay_across_capacity(monkeypatch):
    from scripts import dagger_rl_r3_train as prior,dagger_rl_r4_train as candidate
    monkeypatch.setattr(candidate,'MEMORY_STRATEGY','fifo')
    old=new=None
    for u in range(1,41):
        n=256;ids=np.arange((u-1)*n,u*n).reshape(8,4,8);item=ids%3==0
        labels=np.zeros((8,4,8,5,3),bool);flat=labels.reshape(-1,5,3)
        flat[np.arange(n),ids.reshape(-1)%5,np.where(item.reshape(-1),2,0)]=True
        batch=(ids[...,None],ids[...,None]*2,ids,ids,ids,ids,np.ones((*ids.shape,5),bool),np.ones((*ids.shape,3),bool),labels,item,np.ones_like(item))
        old,rb,stats=prior.aggregate(batch,old,67,u)
        new,cr,cs=candidate.aggregate(batch,new,67,u)
        for a,b in zip(old,new['rows']):np.testing.assert_array_equal(a,b)
        for a,b in zip(rb,cr):np.testing.assert_array_equal(a,b)
        for key in stats:assert stats[key]==cs[key]


@pytest.mark.parametrize('declined',[False,True])
def test_candidate_cannot_execute_before_scientific_selection(tmp_path,monkeypatch,declined):
    from scripts import dagger_rl_r4_run as run
    monkeypatch.setattr(run,'B',tmp_path)
    if declined:(tmp_path/'SCIENTIFIC_SELECTION.json').write_text('{"execute_memory_test":false}')
    with pytest.raises(AssertionError,match='scientific selection'):run.authorize()


def test_dispatch_targets_own_evaluator_and_report_namespace():
    tree=ast.parse(Path('.hermes/dagger_rl_r4_dispatch.py').read_text())
    commands=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert 'scripts/dagger_rl_r4_cpu.py' in commands
    assert 'scripts.dagger_rl_r4_report' in commands
    assert 'scripts/dagger_rl_r3_cpu.py' not in commands
    assert 'scripts.dagger_rl_r3_report' not in commands
