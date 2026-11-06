"""Candidate B: factual temporal continuation, no synchronous CHAIN payment.
Lazy import installer; pressure_count keeps its old shape and packs two reserved flags.
"""
import inspect
VERSION='cx-continuation-v2'
PACK=64
FLAG_BASE=4096
FRONTIER_SEEN=4096
CONCESSION_SEEN=8192
COEF=.01
GAP=24
MAX_CONTINUATIONS=20
SOURCES={}

def continuation_step(count,last_t,last_batch,t,sourcebatch,effective,bad):
 import jax.numpy as j
 total=count%PACK;streak=(count//PACK)%PACK;flags=(count//FLAG_BASE)*FLAG_BASE
 blocked=(count&CONCESSION_SEEN)!=0
 distinct=(sourcebatch>last_batch)&(sourcebatch>=0)&(sourcebatch<(1<<27))&(t>last_t)
 event=effective&distinct&~bad&~blocked
 linked=event&(streak>0)&(t-last_t<=GAP)&(total<MAX_CONTINUATIONS)
 next_streak=j.where(event,j.where((streak>0)&(t-last_t<=GAP),j.minimum(streak+1,31),1),streak)
 pay=j.where(linked,COEF*j.minimum(j.maximum(next_streak-1,0),4),0.)
 # Reset temporal continuity after bad, but never restore the per-episode allowance/seen flags.
 next_streak=j.where(bad,0,next_streak)
 return (pay,flags+total+linked.astype(j.int32)+PACK*next_streak,
         j.where(bad,-1000,j.where(event,t,last_t)),j.where(event,sourcebatch,last_batch),next_streak,linked)

def zero_sync_chain(k,**kwargs):
 import jax.numpy as j
 return j.zeros_like(k,dtype=j.float32)

def install(R=None):
 if R is None:from scripts import r3_items_reward as R
 if getattr(R,'_cx_continuation_v2',False):return R
 from scripts import cc_fountain_reward_v1 as F
 F.install(R)
 scope=R.__dict__;scope.update(fountain_step=continuation_step,chain_reward=zero_sync_chain)
 src=F.SOURCES['transition']
 # No home stacking exception, including enemy visits to our home.
 old="source_valid=resolved_own&((homefields[rows,seats]>3)|(enemy_home[:,None,None]<=3))"
 assert src.count(old)==1
 src=src.replace(old,"source_valid=resolved_own&(homefields[rows,seats]>3)")
 anchor='    s=batch_update(s,before,after,info)'
 assert src.count(anchor)==1
 src=src.replace(anchor,"    metrics.update(r3_f2_continuation_candidate=fpay,r3_f2_continuation_event=pressure,r3_f2_no_sync_chain=j.ones(n,j.int32))\n"+anchor)
 exec(compile(src,'cx_fountain_continuation_v2.transition','exec'),scope);R.transition=scope['transition']
 import jax.numpy as j
 R.CAPS=j.array((.36,.60,.36,.20,.20,0.,.40,.45))
 R._cx_continuation_v2=True
 SOURCES.update(F.SOURCES,transition=src,continuation_step=inspect.getsource(continuation_step),chain_reward=inspect.getsource(zero_sync_chain))
 return R
