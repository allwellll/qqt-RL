import ast
from pathlib import Path
import numpy as np
import jax,jax.numpy as j


def test_item_events_removed_from_attack_cap_but_item_ledger_paid_separately():
    from scripts.bomb_threat_r1_progress_reward import state,settle
    e=dict(active=j.ones(1,bool),bad=j.zeros(1,bool),terminal=j.zeros(1,bool),settled=j.ones(1,bool),placed=j.zeros(1,bool),safe=j.ones(1,bool),threat=j.zeros(1,bool),enemy_cell=j.zeros(1,j.int32),old_ids=j.full((1,195),-1,j.int32),owned_removed=j.zeros((1,13,15),bool),improved=j.zeros(1,bool),multibatch=j.zeros(1,bool),batch_id=j.zeros(1,j.int32),pickup_types=j.ones((1,3),bool),phi_before=j.zeros(1),phi_after=j.zeros(1),phase_before=j.zeros(1,bool),phase_after=j.zeros(1,bool),danger_after=j.zeros(1,bool))
    out,pay,m=settle(e,state(1));np.testing.assert_array_equal(pay,0)
    assert float(out['gross'][0])==0


def test_worker_stages_migration_and_eager_safety_before_pmap():
    from scripts.bomb_threat_r1_progress_train import worker_source
    s=worker_source();compile(s,'new_native_worker','exec')
    assert s.index('native_safety()')<s.index('par,adm=rep(params),rep(adam)')
    assert "params=P.one(par),adam=P.one(adm),states=jax.device_get(st)" in s
    assert "Adam_exact=True" in s and 'old_env_aux_imported=False' in s
    assert "bs,ba=None,None" in s
    assert "bs,ba=M.banks(bs,ba,bm)" in s
    assert 'source_next_step' not in s


def test_owner_consumes_new_gate_before_generic_a3sr_or_failure():
    s=Path('scripts/bomb_threat_owner.py').read_text();ast.parse(s)
    assert s.index('replacement=item_switch')<s.index('replacement=a3sr_switch')
    assert 'b\'scripts.bomb_threat_r1_progress_train\'' in s
    assert "and not entry.get('r1_items_applied')" in s


def test_new_bank_never_imports_old_anonymous_inventory():
    from scripts import bomb_threat_r1_progress_collector as M
    assert "credit['items']" in M._source
    assert "scale=ax['combat_mix']['bomb_threat']['items']['scale']" in M._source
    s=Path('scripts/bomb_threat_r1_items_bank.py').read_text()
    assert "st,ax=E.initial(keys,0)" in s
    assert "pickle.load((OUT/'raw_bank.pkl').open('rb'))" in s
    assert 'p1_bank_exact.pkl' not in s and 'state.pkl' not in s


def test_cuda_environment_selected_before_native_rule_import():
    source=Path('scripts/bomb_threat_r1_progress_train.py').read_text().split('def main():',1)[1]
    assert source.index("JAX_PLATFORMS='cuda'")<source.index('from scripts.bomb_threat_r1_items_runtime import install')
