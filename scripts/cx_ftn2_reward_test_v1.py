"""CPU unit and adversarial tests: real pure settlement + native safety integration."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1')
import inspect,json,time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from scripts import cx_fountain_continuation_v2 as B,cx_safe_frontier_v1 as F,cx_ftn2_reward_v1 as V
from scripts import r3_items_reward as R
from scripts.population_run_state import atomic_json,sha
OUT=Path('runs/cc_train_20261105/training/ftn2_20261106/tests');OUT.mkdir(parents=True,exist_ok=True)
tests={}
def check(name,ok,detail=''):
 ok=bool(ok);tests[name]=dict(passed=ok,detail=str(detail));assert ok,(name,detail)
def near(a,b):return bool(j.allclose(a,b,atol=1e-6))
def seq(events):
 c=j.int32(0);lt=j.int32(-1000);lb=j.int32(-1);out=[]
 for t,b,e,bad in events:
  pay,c,lt,lb,st,hit=B.continuation_step(c,lt,lb,j.int32(t),j.int32(b),j.bool_(e),j.bool_(bad));out.append(float(pay))
 return out,c,lt,lb
p,c,_,_=seq([(0,0,True,False),(10,1,True,False),(20,2,True,False),(30,3,True,False),(40,4,True,False),(50,5,True,False)])
check('B_first_only_seeds_no_payment',p[0]==0,p);check('B_distinct_tick_continuations',near(j.array(p),j.array([0,.01,.02,.03,.04,.04])),p)
p,_,_,_=seq([(0,0,True,False),(0,1,True,False),(10,0,True,False),(12,1,True,False),(15,0,True,False),(20,2,True,False)])
check('B_same_tick_same_batch_old_batch_no_pay',near(j.array(p),j.array([0,0,0,.01,0,.02])),p)
p,_,_,_=seq([(0,0,True,False),(24,1,True,False),(49,2,True,False),(50,3,True,False)])
check('B_gap24_continues_25_reseeds',near(j.array(p),j.array([0,.01,0,.01])),p)
p,_,_,_=seq([(0,0,True,False),(10,1,True,False),(20,2,True,True),(21,3,True,False),(25,4,True,False)])
check('B_bad_event_zero_then_requires_new_seed',near(j.array(p),j.array([0,.01,0,0,.01])),p)
p,_,_,_=seq([(i*8,i,False,False) for i in range(100)]);check('B_empty_far_home_enemy_dead_effective_false_zero',sum(p)==0)
p,c,_,_=seq([(i*8,i,True,False) for i in range(100)]);check('B_per_episode20_continuation_cap',sum(x>0 for x in p)==20 and int(c)%64==20)
check('B_zero_sync_chain_even_k100',near(B.zero_sync_chain(j.arange(100)),0))
count=j.int32(B.FRONTIER_SEEN|B.CONCESSION_SEEN)
check('B_packed_flags_survive_bad',int(B.continuation_step(count,j.int32(1),j.int32(1),j.int32(2),j.int32(2),j.bool_(True),j.bool_(True))[1])==int(count))
kw=dict(c_eligible=True,strict_crossing=True,selected_survivable=True,post_safe=True,moved=True,ability_zero=True,bad=False,terminal=False,enemy_alive=True,home_stack=False,carrying=False,has_item=False,concession=False)
pay,count,hit=F.event_gate(j.int32(0),**kw);check('F_once_safe_crossing',near(pay,.03) and bool(hit))
check('F_same_state_reentry_no_second_event',float(F.event_gate(count,**kw)[0])==0)
for name in ('c_eligible','strict_crossing','selected_survivable','post_safe','moved','ability_zero','enemy_alive'):
 check('F_require_'+name,float(F.event_gate(j.int32(0),**(kw|{name:False}))[0])==0)
for name in ('bad','terminal','home_stack','carrying','has_item','concession'):
 check('F_reject_'+name,float(F.event_gate(j.int32(0),**(kw|{name:True}))[0])==0)
check('F_concession_flag_cannot_reset_on_reentry',float(F.event_gate(j.int32(B.CONCESSION_SEEN),**kw)[0])==0)
V.install(R);check('installed_caps_only_attack_slots_changed',near(R.CAPS[:5],j.array([.36,.6,.36,.2,.2])) and near(R.CAPS[5],.03))
s=R.state(1);candidate=j.zeros((1,8)).at[0,5].set(.03)
for tick in range(30):s,p,m=R.settle(s,candidate if tick==0 else j.zeros((1,8)),bad=j.array([False]),terminal=j.array([False]),active=j.array([True]));check(f'quarantine_no_early_pay_{tick}',float(p.sum())==0)
s,p,m=R.settle(s,j.zeros((1,8)),bad=j.array([False]),terminal=j.array([False]),active=j.array([True]));check('quarantine_pays_at30',near(p[0,5],.03))
s=R.state(1);s,_,_=R.settle(s,candidate,bad=j.array([False]),terminal=j.array([False]),active=j.array([True]));s,p,m=R.settle(s,j.zeros((1,8)),bad=j.array([True]),terminal=j.array([False]),active=j.array([True]));check('own_death_control_voids_pending',float(s['queue'].sum())==0 and near(m['void'].sum(),.03) and float(p.sum())==0)
s=R.state(1);s=dict(s,gross=s['gross'].at[0,5:].set(j.array([.03,.2,.1])),queue=s['queue'].at[0,0,:].set(j.array([.1,.1,0,0,0,.03,.02,.01])))
ss,cc,claw,pending=F.attack_loop_barrier(s,j.ones((1,8)),j.array([True]));check('concession_claws_previous_attack_once',near(claw,.33) and near(cc[0,5],-.33))
check('concession_voids_attack_only_preserves_items',near(ss['queue'][0,0,:5],s['queue'][0,0,:5]) and float(ss['queue'][:,:,5:].sum())==0 and near(pending,.06))
check('clawback_never_refills_cap',near(ss['gross'],s['gross']))
ss2,cc2,claw2,_=F.attack_loop_barrier(ss,j.ones((1,8)),j.array([True]));check('repeated_package_kill_loop_no_extra',near(claw2,0) and float(cc2[:,5:].sum())==0)
from scripts.bun_native_v2_custody_reward import graded_phi
phi=lambda home,carry,loose:float(graded_phi(j.bool_(home),j.bool_(carry),j.bool_(loose),j.float32(.5),j.float32(.5)))
pot=[phi(True,False,False),phi(False,True,False),phi(False,False,True),phi(True,False,False)]
check('release_steal_kill_return_net_nonpositive',sum(b-a for a,b in zip(pot,pot[1:]))+.33-float(claw[0])<=1e-6,dict(phi=pot,paid=.33,claw=float(claw[0])))
# Shape/serialization of packed aux is unchanged; real full collector/Adam fresh is a separate GPU gate.
check('aux_pytree_schema_unchanged',jax.tree.structure(R.state(2))==jax.tree.structure(s))
fork=V.fork_aux(dict(combat_mix=dict(bomb_threat=dict(r3=s))))['combat_mix']['bomb_threat']['r3']
check('fork_keeps_gross_caps_and_item_queue',near(fork['gross'],s['gross']) and near(fork['queue'][:,:,:5],s['queue'][:,:,:5]))
check('fork_voids_legacy_attack_no_pytree_change',float(fork['queue'][:,:,5:].sum())==0 and jax.tree.structure(fork)==jax.tree.structure(s))
check('home_stack_has_no_enemy_visit_exception','source_valid=resolved_own&(homefields[rows,seats]>3)' in B.SOURCES['transition'])
check('native_safety_not_named_survivable_fallback','analyze_selected_actions' in inspect.getsource(F.native_frontier) and 'survivable_bomb_placed' not in inspect.getsource(F.native_frontier))
check('factual_continuation_near_novel_useful_active','effective=explosion&(novel&near).any((-2,-1))' in B.SOURCES['transition'] and 'useful&active&before.alive' in B.SOURCES['transition'])
result=dict(status='PASS',unix=time.time(),cpu_backend=jax.default_backend(),tests=tests,version=V.VERSION,sources={f:sha(f) for f in ('scripts/cx_ftn2_reward_v1.py','scripts/cx_fountain_continuation_v2.py','scripts/cx_safe_frontier_v1.py')},native_integration='SEPARATE_NATIVE_TEST_REQUIRED',loop_scope='same-resource return-to-home custody cycle; all R3 attack paid clawed once, pending voided, no further R3 attack after concession; excludes genuine external item/brick/win events')
atomic_json(OUT/'UNIT_ANTIHACK.json',result);print('PASS',len(tests),'CPU unit/anti-hack tests',flush=True)
if __name__=='__main__':pass
