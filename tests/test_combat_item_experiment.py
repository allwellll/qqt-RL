"""RED: bounded causal item settlement and strict phase gates."""
import numpy as np
import jax
import jax.numpy as j
from scripts import combat_item_events as m


def event(**kw):
 d=dict(banana_start=False,banana_owner=0,glue_hit=False,carry=True,safe=True,bad=False,capture=False,
        sliding=False,slow=False,topology_changed=False,distance=20.,walk_budget=.18,
        chase_delay=0.,legacy_glue=0.,legacy_banana=0.,lost=False)
 d.update(kw);return {k:j.asarray([[v,v]]) for k,v in d.items()}


def test_banana_settles_full_slide_once_and_subtracts_walking_cost():
 s=m.state(1)
 r,s,d=m.settle(event(banana_start=True,sliding=True),s,'banana')
 assert float(r[0,0])==0
 r,s,d=m.settle(event(sliding=True,distance=19.),s,'banana');assert float(r[0,0])==0
 r,s,d=m.settle(event(distance=18.),s,'banana');assert float(r[0,0])>0
 r,s,d=m.settle(event(distance=18.),s,'banana');assert float(r[0,0])==0


def test_short_slide_loiter_harm_and_other_owner_cannot_farm():
 for bad in [dict(banana_owner=1),dict(bad=True),dict(carry=False),dict(topology_changed=True),dict(safe=False)]:
  _,s,_=m.settle(event(banana_start=True,sliding=True,**bad),m.state(1),'banana')
  r,_,_=m.settle(event(distance=19.99),s,'banana');assert float(r[0,0])==0
 _,s,_=m.settle(event(banana_start=True,sliding=True),m.state(1),'banana')
 r,_,_=m.settle(event(distance=19.99),s,'banana');assert float(r[0,0])==0


def test_glue_only_pays_after_real_chase_delay_and_home_progress():
 _,s,_=m.settle(event(glue_hit=True,slow=True),m.state(1),'glue')
 r,s,_=m.settle(event(slow=True,chase_delay=.5,distance=19.),s,'glue');assert float(r[0,0])==0
 r,s,_=m.settle(event(distance=18.),s,'glue');assert float(r[0,0])>0
 r,_,_=m.settle(event(distance=18.),s,'glue');assert float(r[0,0])==0
 for kw in [dict(chase_delay=0.,distance=18.),dict(chase_delay=.5,distance=21.)]:
  _,s,_=m.settle(event(glue_hit=True,slow=True),m.state(1),'glue')
  _,s,_=m.settle(event(slow=True,**kw),s,'glue')
  r,_,_=m.settle(event(distance=kw['distance']),s,'glue');assert float(r[0,0])==0


def test_clawback_cap_reset_and_zero_delta_equivalence():
 s=m.state(1)
 for i in range(8):
  _,s,_=m.settle(event(banana_start=True,sliding=True,distance=30.-3*i),s,'banana')
  _,s,_=m.settle(event(distance=27.-3*i),s,'banana')
 assert float(s['banana_gross'][0,0])<=2
 r,s,_=m.settle(event(bad=True,lost=True,carry=False),s,'banana')
 assert float(r[0,0])==-2 and float(s['banana_balance'][0,0])==0
 base=j.array([[.3,.4]])
 np.testing.assert_array_equal(m.adjust(base,j.ones((1,2)),0),base)


def test_phase_stock_needs_native_window_but_carrier_survives_respawn():
 assert not bool(m.phase_allowed(False,True,0))
 assert bool(m.phase_allowed(False,False,0))
 assert bool(m.phase_allowed(False,True,1))
 assert bool(m.phase_allowed(True,True,0))


def test_finite_zero_and_nonzero_actor_gradient_through_reward_gae():
 from jax_bomb.jax_train import compute_gae
 def f(logit,coefficient):
  extra=m.adjust(j.zeros((2,1)),j.array([[.2],[0.]]),coefficient)
  adv=compute_gae(extra,j.zeros_like(extra),j.zeros_like(extra),j.array([[False],[True]]),.995,.95)
  return -(jax.nn.log_softmax(j.array([logit,0.]))[0]*adv[0,0])
 assert float(jax.grad(f)(0.,0.))==0
 assert abs(float(jax.grad(f)(0.,1.)))>0


def test_real_native_speed_direction_and_collision():
 from jax_bomb import je07_bun_env as env
 env.prepare();key=jax.random.PRNGKey(20261023);st=env.init_batch(key,1)
 st=st._replace(core=st.core._replace(pos=j.array([[[5.5,5.5],[9.5,10.5]]]),wall=j.zeros_like(st.wall),brick=j.zeros_like(st.brick),invuln=j.zeros_like(st.core.invuln)),bun_carried=j.array([[1,-1]],j.int8))
 v=jax.vmap(env._movement_scale)
 np.testing.assert_allclose(np.asarray(v(st)*env.base.STEP*st.core.spd_g)[0,0],.18,rtol=1e-5)
 st=st._replace(move_status=st.move_status.at[0,0].set(2),slide_dir=st.slide_dir.at[0,0].set(3))
 np.testing.assert_allclose(np.asarray(v(st)*env.base.STEP*st.core.spd_g)[0,0],.67,rtol=1e-5)
 step=jax.jit(jax.vmap(lambda s:env.step(s,j.array([[2,0],[4,0]]),key,False,True)))
 after,_,info=step(st)
 assert float(after.pos[0,0,1])>float(st.pos[0,0,1])
 assert int(after.move_status[0,0])==2
 st=st._replace(core=st.core._replace(wall=st.wall.at[0,5,6].set(True),pos=st.pos.at[0,0,1].set(5.69)))
 after,_,_=step(st);assert int(after.move_status[0,0])==0


def test_saved_legacy_family_replacement_refunds_exactly_once():
 _,s,_=m.settle(event(glue_hit=True,slow=True,legacy_glue=.5),m.state(1),'glue')
 # Replace a legacy hit that did not actually delay chase: remove .5 now;
 # on later loss its legacy clawback must be cancelled, not doubled.
 r,s,_=m.settle(event(bad=True,lost=True,carry=False),s,'glue')
 assert np.isclose(float(r[0,0]),.5)
 r,_,_=m.settle(event(bad=True,lost=True,carry=False),s,'glue');assert float(r[0,0])==0


def test_combat_coefficient_zero_exact_and_independent_actor_critic():
 from scripts.combat_item_train import objective
 p=dict(actor=j.zeros(5),critic=j.ones(1))
 def forward(q,arch,o,g):return j.tile(q['actor'],(2,1)),j.zeros((2,3)),j.zeros(2),j.tile(q['critic'],(2,1))
 route=j.zeros((2,5,3),bool);combat=route.at[:,0,0].set(True)
 b=(j.zeros((2,1)),j.zeros((2,1)),j.zeros((2,2),int),j.zeros(2),j.ones(2),j.zeros(2),j.ones((2,5),bool),j.ones((2,3),bool),route,combat)
 f=lambda q,c:objective(q,b,c,forward,lambda v,r:j.mean((v[:,0]-r)**2))[0]
 g0=jax.grad(f)(p,0.);g1=jax.grad(f)(p,.05)
 np.testing.assert_array_equal(g0['critic'],g1['critic'])
 assert np.linalg.norm(g0['actor']-g1['actor'])>0


def test_distinct_item_events_cannot_farm_the_same_return_corridor():
 for family in ['banana','glue']:
  s=m.state(1)
  for repeat in range(2):
   if family=='banana':
    _,s,_=m.settle(event(banana_start=True,sliding=True,distance=20.),s,family)
    r,s,_=m.settle(event(distance=18.),s,family)
   else:
    _,s,_=m.settle(event(glue_hit=True,slow=True,distance=20.),s,family)
    _,s,_=m.settle(event(slow=True,chase_delay=.5,distance=19.),s,family)
    r,s,_=m.settle(event(distance=18.),s,family)
   if repeat==0:assert float(r[0,0])>0
   else:assert float(r[0,0])==0,'new item identity cannot repay an old corridor after walking backwards'
