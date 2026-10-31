import numpy as np
import pytest
from scripts.carrier_mechanism_stats import discounted_returns,calibration,cluster_interval,validate_bank_record,paired_q_contrasts

def test_terminal_has_zero_bootstrap_and_truncation_is_separate():
 r=np.array([[1.,2.,99.],[3.,4.,5.]])
 d=np.array([[False,True,False],[False,False,False]])
 x=discounted_returns(r,d,.5)
 assert x[0]==2 and x[1]==6.25
 assert np.array_equal(discounted_returns(r,d,.5,tail=np.array([900.,8.])),[2.,7.25])

def test_reject_injected_carrier_or_free_alive_stock():
 base=dict(before_carried=-1,after_carried=1,learner_alive_before=True,learner_alive_after=True,stock_steal=True,enemy_alive=False,enemy_trapped=0,source_episode='e',layout='native',split='train',provenance='actual_env_step')
 validate_bank_record(base)
 for bad in [dict(before_carried=1),dict(enemy_alive=True),dict(provenance='preset')]:
  with pytest.raises(ValueError):validate_bank_record(dict(base,**bad))

def test_cluster_ci_does_not_treat_repeats_as_independent_states():
 y=np.array([0]*8+[1]*8,float);ids=['e0']*8+['e1']*8
 assert cluster_interval(y,ids,seed=42)==cluster_interval(y,ids,seed=42)
 lo,hi=cluster_interval(y,ids,seed=42);assert lo==0 and hi==1

def test_calibration_reports_low_target_variance_and_bias():
 m=calibration(np.array([2.,2.]),np.ones((2,8)))
 assert m['explained_variance'] is None and m['bias']==1 and m['rmse']==1

def test_shared_future_q_difference_and_subset_label():
 x=np.array([[3.,4.],[1.,2.],[2.,3.]])
 r=paired_q_contrasts(x,['forward','backward','wait'])
 assert r['estimand']=='subset Q contrasts' and r['forward_minus_backward']['mean']==2


def test_bankP_later_events_exclude_pre_deviation_and_invalid_padding():
 from scripts.carrier_mechanism_stats import event_outcomes_from
 flags=np.zeros((5,5),bool);valid=np.array([True,True,True,True,False])
 flags[0,1]=True;flags[2,2]=True;flags[3,0]=True;flags[4,1]=True
 out=event_outcomes_from(flags,valid,2)
 assert not out['died_later'] and out['dropped_later'] and out['eventual_delivery']
 assert out['immediate_drop'] and not out['immediate_death']
 assert out['recovery_delivery_after_failure']
 out=event_outcomes_from(flags,valid,3)
 assert out['eventual_delivery'] and not out['dropped_later']


def test_calibration_ci_resamples_whole_episode_not_rollout():
 from scripts.carrier_mechanism_stats import calibration_intervals
 v=np.array([0.,2.]);r=np.array([[0.]*8,[1.]*8])
 x=calibration_intervals(v,r,repetitions=1000)
 assert x['bias']==[0.,1.] and x['rmse']==[0.,1.]


def test_terminal_gae_ignores_next_episode_value():
 import jax.numpy as jnp
 from jax_bomb.jax_train import compute_gae
 a=compute_gae(jnp.array([[2.],[4.]]),jnp.array([[1.],[3.]]),jnp.array([[900.],[5.]]),jnp.array([[True],[False]]),.5,.95)
 assert np.isclose(a[0,0],1.) and np.isclose(a[1,0],3.5)


def test_dagger_only_has_no_critic_gradient_and_uses_set_mass():
 import jax,jax.numpy as jnp
 from scripts.carrier_mechanism_train import objective
 p=dict(actor=jnp.array([0.,0.,0.,0.,0.]),critic=jnp.array([1.]))
 def f(q,arch,o,g):return jnp.tile(q['actor'],(2,1)),jnp.zeros((2,3)),jnp.zeros(2),jnp.tile(q['critic'],(2,1))
 b=(jnp.zeros((2,1)),jnp.zeros((2,1)),jnp.zeros((2,2),int),jnp.zeros(2),jnp.ones(2),jnp.zeros(2),jnp.ones((2,5),bool),jnp.ones((2,3),bool),jnp.array([[[True,False,False]]+[[False]*3]*4]*2))
 loss,metrics=objective(p,b,'DAgger-only',f,lambda v,r:jnp.mean((v[:,0]-r)**2))
 grad=jax.grad(lambda q:objective(q,b,'DAgger-only',f,lambda v,r:jnp.mean((v[:,0]-r)**2))[0])(p)
 assert np.allclose(grad['critic'],0) and np.linalg.norm(grad['actor'])>0
 assert np.isclose(float(loss),.2*np.log(15),rtol=1e-5)


def test_dagger_critic_freeze_preserves_nonzero_saved_adam_moments():
 import jax,jax.numpy as jnp,optax
 from scripts.carrier_mechanism_train import preserve_critic
 params=dict(actor=jnp.array([1.]),critic=jnp.array([2.]))
 opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
 previous=opt.init(params)
 _,previous=opt.update(jax.tree.map(jnp.ones_like,params),previous,params)
 changes,after=opt.update(dict(actor=jnp.ones(1),critic=jnp.zeros(1)),previous,params)
 assert float(jnp.linalg.norm(changes['critic']))>0  # zero gradient alone is not a freeze
 preserved=preserve_critic(after,previous)
 for name in ['mu','nu']:
  np.testing.assert_array_equal(getattr(preserved[1][0],name)['critic'],getattr(previous[1][0],name)['critic'])
 assert int(preserved[1][0].count)==int(previous[1][0].count)+1


def test_fixed_cpu_episode_shards_reject_duplicates_or_model_mixing():
 from scripts.retention_route_eval import merge_episodes
 def shard(game,model='same'):
  e=dict(game=game,seed=100+game,team=game%2,pickup=1,delivery=0,oracle_opportunities=2,first_oracle_match=1,stage_funnel={})
  return dict(passed=True,model_json_sha256=model,seeds=[100+game],results={'natural_full2400':dict(games=1,episodes=[e])})
 good=merge_episodes('natural_full2400',[shard(1),shard(0)])
 assert [e['game'] for e in good['results']['natural_full2400']['episodes']]==[0,1]
 with pytest.raises(ValueError):merge_episodes('natural_full2400',[shard(0),shard(0)])
 with pytest.raises(ValueError):merge_episodes('natural_full2400',[shard(0),shard(1,'other')])


def test_natural_report_separates_episode_events_and_weighted_tick_ratios():
 from scripts.carrier_mechanism_report import natural_observation
 episodes=[dict(game=i,seed=10+i,team=i,delivery=2 if i==0 else 0,
     oracle_matches=1,oracle_opportunities=2+6*i,first_oracle_matches=1,
     first_oracle_actions=1,progress_ticks=1,target_ticks=2,stage_funnel={}) for i in range(2)]
 out,binary=natural_observation({'results':{'natural_full2400':{'episodes':episodes}}})
 assert out['episode_counts']['delivery']==1 and binary['delivery'].tolist()==[1,0]
 assert out['tick_oracle_agreement']==.2 and out['progress_per_target_tick']==.5


def test_parallel_cpu_dispatch_keeps_shared_status_single_writer(tmp_path,monkeypatch):
 import threading
 from types import SimpleNamespace
 from scripts import carrier_mechanism_cpu as cpu
 from scripts.population_run_state import atomic_json
 monkeypatch.setattr(cpu,'R',tmp_path);monkeypatch.setattr(cpu,'B',tmp_path/'run');cpu.B.mkdir()
 owner_thread=threading.get_ident();barrier=threading.Barrier(4)
 def guarded_write(path,value):
  assert threading.get_ident()==owner_thread,'parallel workers must not write shared status/temp files'
  atomic_json(path,value)
 monkeypatch.setattr(cpu,'atomic_json',guarded_write)
 def fake_run(*args,**kwargs):
  barrier.wait(timeout=3);return SimpleNamespace(returncode=0)
 monkeypatch.setattr(cpu.subprocess,'run',fake_run)
 pending=[tmp_path/f'model{i}' for i in range(4)]
 assert cpu.dispatch_pending(pending)==[(p,0) for p in pending]
