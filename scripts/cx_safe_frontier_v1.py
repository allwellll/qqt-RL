"""Candidate 2: one native-certified c-gated frontier crossing per episode.
No action-count/stock-tick reward. Existing-pytree packed flags survive respawns.
"""
import inspect
from scripts.cx_fountain_continuation_v2 import FRONTIER_SEEN,CONCESSION_SEEN
EVENT=.03
VERSION='cx-safe-frontier-v1'
SOURCES={}

def event_gate(count,*,c_eligible,strict_crossing,selected_survivable,post_safe,moved,ability_zero,bad,terminal,enemy_alive,home_stack,carrying,has_item,concession):
 import jax.numpy as j
 available=((count&FRONTIER_SEEN)==0)&((count&CONCESSION_SEEN)==0)
 hit=available&c_eligible&strict_crossing&selected_survivable&post_safe&moved&ability_zero&~bad&~terminal&enemy_alive&~home_stack&~carrying&~has_item&~concession
 return j.where(hit,EVENT,0.),count|j.where(hit,FRONTIER_SEEN,0).astype(j.int32),hit

def attack_loop_barrier(s,candidate,concession):
 """First enemy package acquisition cancels ALL unpaid attack and refunds no cap.
 Previously paid R3 attack is clawed back exactly once; no subsequent positive attack.
 Item/road queues and payment semantics are preserved. Full custody loop telescopes.
 """
 import jax.numpy as j
 old=(s['pressure_count']&CONCESSION_SEEN)!=0;first=concession&~old;blocked=old|concession
 claw=j.where(first,s['gross'][:,5:].sum(-1),0.)
 pending=j.where(blocked[:,None,None],s['queue'][:,:,5:],0.)
 queue=s['queue'].at[:,:,5:].set(j.where(blocked[:,None,None],0.,s['queue'][:,:,5:]))
 void=s['void'].at[:,5:].add(pending.sum(1))
 c=candidate.at[:,5:].set(j.where(blocked[:,None],0.,candidate[:,5:]));c=c.at[:,5].add(-claw)
 count=s['pressure_count']|j.where(blocked,CONCESSION_SEEN,0).astype(j.int32)
 return dict(s,queue=queue,void=void,pressure_count=count),c,claw,pending.sum((1,2))

def attack_positions(st,targets,seats):
 """Walkable cells whose current native blast range reaches a frozen enemy/brick.
 Stop at walls, bricks, pushable objects and live bombs. Never count walls as targets.
 """
 import jax.numpy as j
 solid=st.wall|st.brick|st.core.pushable|(st.fuse>1)
 goals=j.zeros_like(st.brick);cap=st.core.blast_cap[j.arange(len(seats)),seats]
 def shifted(x,dr,dc):
  z=j.pad(x,((0,0),(1,1),(1,1)),constant_values=False)
  return z[:,1+dr:14+dr,1+dc:16+dc]
 for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
  ray=targets
  for k in range(1,9):
   ray=shifted(ray,dr,dc)
   goals|=ray&~solid&(cap[:,None,None]>=k)
   ray&=~solid
 return goals

def native_frontier(before,after,info,aux,seats,homefields,count):
 """Actual joint selected-action safety + after-state threat crossing, never legal==safe."""
 import jax
 import jax.numpy as j
 from scripts.carrier_mechanism_engine import env
 from scripts import r3_items_reward as R
 from scripts.bomb_threat_r1_progress_reward import native_safety
 n=len(seats);rows=j.arange(n);enemy=1-seats
 clip=lambda x:j.clip(j.floor(x).astype(j.int32),j.array([0,0]),j.array([12,14]))
 start=clip(before.pos[rows,seats]);end=clip(after.pos[rows,seats]);ec=clip(before.pos[rows,enemy]);ea=clip(after.pos[rows,enemy])
 act=info['h_joint_actions'][rows,seats]
 mm,bm=jax.vmap(env.legal_mask)(before);chosen_legal=mm[rows,seats,act[:,0]]&bm[rows,seats,act[:,1]]
 live=((before.fuse>0)&(before.core.owner==seats[:,None,None])).sum((-2,-1))
 carrying=(before.bun_carried[rows,seats]>=0)|(after.bun_carried[rows,seats]>=0)
 has_item=(before.item_counts[rows,seats].sum(-1)>0)|(after.item_counts[rows,seats].sum(-1)>0)
 enemy_alive=before.alive[rows,enemy]&after.alive[rows,enemy]&(before.trapped[rows,enemy]==0)&(after.trapped[rows,enemy]==0)
 active=~aux['terminal']&~aux['r1_next_terminal']&before.alive[rows,seats]&(before.trapped[rows,seats]==0)&after.alive[rows,seats]&(after.trapped[rows,seats]==0)
 moved=j.abs(after.pos[rows,seats]-before.pos[rows,seats]).sum(-1)>1e-6
 home_stack=(homefields[rows,seats,start[:,0],start[:,1]]<=3)|(homefields[rows,seats,end[:,0],end[:,1]]<=3)
 eligible=chosen_legal&((count&(FRONTIER_SEEN|CONCESSION_SEEN))==0)&(before.bun_carried[rows,enemy]<0)&(after.bun_carried[rows,enemy]<0)&active&enemy_alive&~carrying&~has_item&~home_stack&(act[:,1]==0)&(act[:,0]<4)&moved&(live<before.core.bombs_cap[rows,seats])&(before.move_status[rows,seats]==0)
 def measure(_):
  frozen_enemy=j.zeros_like(before.brick).at[rows,ec[:,0],ec[:,1]].set(True)
  goals=attack_positions(before,before.brick|frozen_enemy,seats)
  # Enemy movement alone, vanished bricks, newly extended range, new spawn are not progress.
  actual_enemy=frozen_enemy&(ec==ea).all(-1)[:,None,None]
  goals_after=attack_positions(after,(before.brick&after.brick)|actual_enemy,seats)
  blocked_state=before._replace(core=before.core._replace(brick=before.brick|(before.fuse>1)))
  field=R.shortest_fields(blocked_state,goals[:,None])[:,0]
  d0=field[rows,start[:,0],start[:,1]];d1=field[rows,end[:,0],end[:,1]]
  crossing=(d0>0)&(d0<10000)&(d1==0)&goals[rows,end[:,0],end[:,1]]&goals_after[rows,end[:,0],end[:,1]]
  def certify(_):
   safety=native_safety()
   return jax.vmap(lambda st,a:safety.analyze_selected_actions(st,a).selected_survivable)(before,info['h_joint_actions'])[rows,seats]
  selected=jax.lax.cond(j.sum(eligible&crossing,dtype=j.int32)>0,certify,lambda _:j.zeros(n,bool),None)
  danger0=jax.vmap(env.base._danger_map)(before.fuse,before.wall,before.core.bomb_blast,before.brick)
  danger1=jax.vmap(env.base._danger_map)(after.fuse,after.wall,after.core.bomb_blast,after.brick)
  safe0=(danger0[rows,start[:,0],start[:,1]]<=0)&(before.core.blast_linger[rows,start[:,0],start[:,1]]<=0)
  safe1=(danger1[rows,end[:,0],end[:,1]]<=0)&(after.core.blast_linger[rows,end[:,0],end[:,1]]<=0)
  # Existing danger on any cell overlapped by the swept character footprint rejects the event.
  rr,cc=j.indices((13,15));rad=env.base.RADIUS
  lo=j.minimum(before.pos[rows,seats],after.pos[rows,seats])-rad;hi=j.maximum(before.pos[rows,seats],after.pos[rows,seats])+rad
  swept=(rr[None]>=j.floor(lo[:,0,None,None]))&(rr[None]<=j.floor(hi[:,0,None,None]))&(cc[None]>=j.floor(lo[:,1,None,None]))&(cc[None]<=j.floor(hi[:,1,None,None]))
  sweep_safe=~(((danger0>0)|(before.core.blast_linger>0))&swept).any((-2,-1))
  return crossing,selected,safe0&safe1&sweep_safe
 zero=lambda _:(j.zeros(n,bool),j.zeros(n,bool),j.zeros(n,bool))
 crossing,selected,post=jax.lax.cond(j.sum(eligible,dtype=j.int32)>0,measure,zero,None)
 return dict(c_eligible=eligible,strict_crossing=crossing,selected_survivable=selected,post_safe=post,moved=moved,ability_zero=act[:,1]==0,enemy_alive=enemy_alive,home_stack=home_stack,carrying=carrying,has_item=has_item)

def install(R=None):
 if R is None:from scripts import r3_items_reward as R
 if getattr(R,'_cx_frontier_v1',False):return R
 assert getattr(R,'_cx_continuation_v2',False),'requires explicit packed-count ABI from continuation v2'
 original=R.transition
 def transition(before,after,info,aux,metadata,seats):
  import jax.numpy as j
  s,c,metrics,road,bad,terminal,active=original(before,after,info,aux,metadata,seats)
  rows=j.arange(len(seats));enemy=1-seats
  concession=(before.bun_carried[rows,enemy]>=0)|(after.bun_carried[rows,enemy]>=0)|info['steal'][rows,enemy]|info['pickup_loose'][rows,enemy]
  from scripts import bun_native_v2_custody_reward as custody
  cells=j.clip(j.floor(after.pos).astype(j.int32),j.array([0,0]),j.array([12,14]))
  from scripts.carrier_mechanism_engine import env
  cages=j.stack([env._base_mask(0),env._base_mask(1)])
  homefields=R.shortest_fields(after,j.broadcast_to(cages,(len(seats),2,13,15)))
  proof=native_frontier(before,after,info,aux,seats,homefields,s['pressure_count'])
  pay,count,hit=event_gate(s['pressure_count'],**proof,bad=bad,terminal=terminal,concession=concession)
  s=dict(s,pressure_count=count);c=c.at[:,R.CHAIN].add(pay)
  # Suppress every defensive R3 attack, including old THREAT; match custody's victim rule.
  defensive=custody.victim_factors(before)[rows,seats]<=0
  c=c.at[:,5:].set(j.where(defensive[:,None],0.,c[:,5:]))
  s,c,claw,pending=attack_loop_barrier(s,c,concession)
  metrics.update(r3_f2_variant=j.full(len(seats),2,j.int32),r3_f2_frontier_event=hit&(c[:,R.CHAIN]>0),r3_f2_frontier_candidate=j.maximum(c[:,R.CHAIN],0.),
    r3_f2_continuation_candidate_raw=metrics['r3_f2_continuation_candidate'],r3_f2_continuation_candidate=j.maximum(c[:,R.PRESSURE],0.),
    r3_f2_c_eligible=proof['c_eligible'],r3_f2_selected_safe=proof['selected_survivable'],r3_f2_post_safe=proof['post_safe'],
    r3_f2_strict_crossing=proof['strict_crossing'],r3_f2_concession=concession,r3_f2_clawback=claw,r3_f2_attack_pending_void=pending,
    r3_event_candidate=c)
  return s,c,metrics,road,bad,terminal,active
 R.transition=transition
 import jax.numpy as j
 R.CAPS=R.CAPS.at[R.CHAIN].set(EVENT)
 R._cx_frontier_v1=True
 SOURCES.update(transition=inspect.getsource(transition),event_gate=inspect.getsource(event_gate),attack_loop_barrier=inspect.getsource(attack_loop_barrier),native_frontier=inspect.getsource(native_frontier),attack_positions=inspect.getsource(attack_positions))
 return R
