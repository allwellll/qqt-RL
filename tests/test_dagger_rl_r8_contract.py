"""R8 keeps common initialization/data/physics and varies supervised weight only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_registered_single_variable_and_fixed_paired_budget():
    b=ROOT/'runs/dagger_rl_20261024'
    old=json.loads((b/'r6_opponent_candidate/PROTOCOL.json').read_text())
    p=json.loads((b/'r8_dagger_scale/PROTOCOL.json').read_text())
    assert p['arms']==['control','sup_quarter']
    assert p['dagger_coefficient_by_arm']==dict(control=.2,sup_quarter=.05)
    for k in ['parent','parent_input_sha256','optimizer','train_mixture','bank_source','bank_sha256',
              'oracle','aggregation','envs','steps','updates','checkpoints','lr','gamma','gae_lambda',
              'clip','value','entropy','gradient_clip','epochs','minibatch','anchor_coefficient']:
        assert p[k]==old[k],k
    assert p['total_environment_steps']==6*32*8192
    assert p['total_optimizer_steps']==6*32*32
    assert len(set(p['paired_seeds']))==3
    assert set(p['paired_seeds']).isdisjoint(old['paired_seeds'])
    assert p['opponents']['control']==p['opponents']['sup_quarter']
    assert p['opponents']['current']==p['opponents']['ring']==0
    assert p['opponents']['fixed']==1
    a=p['evaluation']['natural_holdout_seed'];z=old['evaluation']['natural_holdout_seed']
    assert set(range(a,a+64)).isdisjoint(range(z,z+64))


def test_actual_collector_and_reference_objective_are_reused():
    from scripts import dagger_rl_r8_train as new,dagger_rl_r6_train as old
    assert new.collector is old.collector
    assert new.aggregate is old.aggregate
    assert new.ORIGINAL_OBJECTIVE is old.C.objective
