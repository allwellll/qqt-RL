"""Isolated-process reward wiring: old workers must stay byte-stable."""
import os
import subprocess
import sys


def test_route_and_kill_discount_delayed_settlement_and_clawback():
    code=r'''
import jax,jax.numpy as j,numpy as np
from scripts.bun_native_v2_runtime import install
install()
from scripts import carrier_mechanism_engine as E
E.configure()
from scripts import bun_native_v2_reward as V,je04_bun_ops as L
from scripts import je09_route_pbrs as P
from scripts.population_run_state import atomic_json
from pathlib import Path
keys=jax.random.split(jax.random.PRNGKey(2),1)
s,aux=E.initial(keys,0)
# Deliberately close, alive and empty-handed: old outer gate would zero PBRS.
s=s._replace(core=s.core._replace(pos=j.array([[[2.5,5.5],[2.5,8.5]]])),bun_carried=j.array([[-1,-1]]),trapped=j.array([[0,0]]))
assert np.allclose(np.asarray(V.victim_factors(s)),.3)
V.install(True)
route=dict(aux['routing'],stock_distance=j.ones((1,2,13,15),j.int32)*5)
phase=P.phase_values(s,route,aux['drop'])
assert np.asarray(phase[0]).tolist()==[[1,1]],'free enemy still suppresses stock route'
assert np.all(np.asarray(phase[2])<0),'route potential still gated off'
carry=s._replace(bun_carried=j.array([[1,0]]))
phase=P.phase_values(carry,route,aux['drop'])
assert np.all(np.asarray(phase[0])>=3),'carrying must target home after enemy revival'
# Real native inventory pickup, free/alive enemy: verify the actual composition
# consumes the new legal-steal event rather than just changing PBRS labels.
stock=s._replace(core=s.core._replace(pos=j.array([[[2.5,9.5],[10.5,3.5]]]),brick=j.zeros_like(s.brick),pushable=j.zeros_like(s.core.pushable)))
after,na,total,done,real_info,parts=jax.jit(E.transition)(stock,aux,j.array([[[4,0],[4,0]]]),keys)
assert bool(real_info['steal'][0,0])
assert abs(float(parts[0,0,E.ops.COMPONENTS.index('transport_shape')])-.75)<1e-5,'legal stock .75 remained window-gated'
home=stock._replace(core=stock.core._replace(pos=stock.pos.at[0,0].set(j.array([2.5,5.5]))),bun_carried=j.array([[1,-1]]),bun_stored=stock.bun_stored.at[0,1,1].set(0))
_,_,_,_,delivery_info,delivery_parts=jax.jit(E.transition)(home,aux,j.array([[[4,0],[4,0]]]),keys)
assert bool(delivery_info['capture'][0,0])
assert abs(float(delivery_parts[0,0,E.ops.COMPONENTS.index('transport_capture')])-24.)<1e-5,'actual terminal delivery24 changed'
names=set(L.TASK_EVENTS)|set(('death','mutual_death','threat_bomb_placed','safe_bomb_placed','tactical_bomb_enemy_safe_moves_before','tactical_bomb_enemy_safe_moves_after','danger_safe_resolution','surviving_kill','own_bomb_defeat','avoidable_danger_death','drop'))
info={k:j.zeros((1,2),bool) for k in names};info['mutual_death']=j.zeros(1,bool)
for k in ('tactical_bomb_enemy_safe_moves_before','tactical_bomb_enemy_safe_moves_after'):info[k]=j.zeros((1,2),j.int32)
info['surviving_kill']=j.array([[True,False]]);info['v2_kill_factor']=j.array([[.3,1.]])
a=L.reward_state(1)
r,a,parts=L.retention_reward(info,a,j.array([False]))
assert float(r[0,0])==0
assert abs(float(a['pending_kill'][0,0])-4.8)<1e-5
info['surviving_kill']=j.zeros((1,2),bool)
r,a,parts=L.retention_reward(info,a,j.array([True]))
assert abs(float(r[0,0])-4.8)<1e-5
info['death']=j.array([[True,False]])
r,a,parts=L.retention_reward(info,a,j.array([True]))
assert abs(float(parts[0,0,4])+4.8)<1e-5,'clawback must reclaim discounted actual paid amount'
# Capped delivery TASK reward is unaffected by the factor.
info['death']=j.zeros((1,2),bool);info['capture']=j.array([[True,False]])
r,a,parts=L.retention_reward(info,L.reward_state(1),j.array([True]))
assert abs(float(r[0,0])-2.)<1e-5
atomic_json(Path('runs/bun_native_v2_20261030/REWARD_WIRING.json'),dict(status='PASS',free_enemy_stock_phase=True,free_enemy_legal_stock_075_in_real_composition=True,carry_enemy_revival_home=True,kill_accrued=4.8,kill_full=16,delayed_settlement=True,clawback_exact=True,delivery_unscaled=True))
'''
    subprocess.run([sys.executable,'-c',code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))


def test_generated_worker_and_collector_compile():
    from scripts.bun_native_v2_train import worker_source
    s=worker_source()
    compile(s,'<native-v2-worker>','exec')
    assert "M.configure(protocol['r1_attack'],protocol['r2_antifarm'])" in s
    assert "rule_version='bun-native-v2'" in s
    assert "ROOT/'runs/bomb_threat_v1'" in s
    assert 'STOP_FOR_REBOOT.json' in s


def test_collector_enemy_stock_failure_and_parts_actual_reward():
    code=r'''
import jax,jax.numpy as j,numpy as np
from scripts.bun_native_v2_runtime import install
install()
from scripts import bun_native_v2_collector as M
M.E.configure();M.configure(False,True)
keys=jax.random.split(jax.random.PRNGKey(72),2)
s,a=M.E.initial(keys,0)
# Each learner's enemy physically enters its real stocked home cage. No fake
# carrier is supplied; env.step must perform the native legal stock pickup.
s=s._replace(core=s.core._replace(pos=j.array([[[10.5,3.5],[2.5,5.5]],[[2.5,9.5],[10.5,10.5]]]),
    brick=j.zeros_like(s.brick),pushable=j.zeros_like(s.core.pushable)))
a=dict(a,tactics=M._module.original.T.events.state(2),provenance_source=j.full(2,-1,j.int32),
    r1_attribute_highwater=j.zeros((2,2,3)),a3sr={k:j.zeros(2) for k in ('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')})
a=dict(a,combat_mix=M.metadata(s,j.array([0,0]),j.array([False,False]),j.array([0,1])))
step=jax.jit(M.transition)
after,na,r,d,info,parts,metrics=step(s,a,j.tile(j.array([[[4,0],[4,0]]]),(2,1,1)),keys,0.)
np.testing.assert_array_equal(metrics['v2_enemy_stock_pickups'],[1,1])
np.testing.assert_allclose(metrics['v2_defense_failure_pay'],[-.05,-.05],rtol=0,atol=1e-7)
np.testing.assert_array_equal(parts[j.arange(2),j.array([0,1]),-1],r[j.arange(2),j.array([0,1])])
after,na,r,d,info,parts,metrics=step(after,na,j.tile(j.array([[[4,0],[4,0]]]),(2,1,1)),keys,0.)
np.testing.assert_array_equal(metrics['v2_enemy_stock_pickups'],[0,0])
np.testing.assert_array_equal(metrics['v2_defense_failure_pay'],[0,0])
from scripts.population_run_state import atomic_json
from pathlib import Path
atomic_json(Path('runs/bun_native_v2_20261030/COLLECTOR_WIRING.json'),dict(status='PASS',both_physical_seats=True,real_native_enemy_pickup=True,per_pickup_penalty=-.05,no_tick_repeat=True,parts_total_equals_collector_reward=True))
'''
    subprocess.run([sys.executable,'-c',code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
