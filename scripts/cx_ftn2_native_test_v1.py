"""CPU real native-state certification and shape/JIT tests for FTN2 frontier."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
import time
import jax
import jax.numpy as j
from scripts import cx_ftn2_reward_v1 as V,cx_safe_frontier_v1 as F
from scripts.population_run_state import atomic_json,sha
from pathlib import Path
OUT=Path('runs/cc_train_20261105/training/ftn2_20261106/tests');OUT.mkdir(parents=True,exist_ok=True)
tests={}
def check(k,v):tests[k]=bool(v);assert bool(v),k
V.install()
from scripts.carrier_mechanism_engine import env
from scripts.bomb_threat_r1_progress_reward import native_safety
from scripts import r3_items_reward as R
env.prepare();env.configure_training('full=1',1,reward_profile='legacy',retention_replay=0,tactical_bomb_placement_reward=.1)
st=jax.jit(lambda key:env.init_batch(key,1,jax.nn.one_hot(0,16)))(jax.random.PRNGKey(20261106));jax.block_until_ready(st)
print('native fields',st._fields,st.core._fields,flush=True)
core=st.core._replace(wall=j.zeros_like(st.wall),brick=j.zeros_like(st.brick),pushable=j.zeros_like(st.core.pushable),fuse=j.zeros_like(st.fuse),owner=j.full_like(st.core.owner,-1),blast_linger=j.zeros_like(st.core.blast_linger),
 pos=j.array([[[6.5,7.9],[6.5,10.5]]],j.float32),alive=j.ones_like(st.alive),bombs_cap=j.full_like(st.core.bombs_cap,2),blast_cap=j.full_like(st.core.blast_cap,2))
st=st._replace(core=core,bun_carried=j.full_like(st.bun_carried,-1),trapped=j.zeros_like(st.trapped),item_counts=j.zeros_like(st.item_counts),move_status=j.zeros_like(st.move_status))
actions=j.array([[[3,0],[4,0]]],j.int32)
step=jax.jit(lambda s,a,k:jax.vmap(lambda x,y,z:env.step(x,y,z,auto_reset=False,return_info=True))(s,a,k))
after,done,info=step(st,actions,jax.random.split(jax.random.PRNGKey(43),1));jax.block_until_ready(after)
info=dict(info,h_joint_actions=actions);seats=j.array([0]);aux=dict(terminal=j.array([False]),r1_next_terminal=j.array([False]));cages=j.stack([env._base_mask(0),env._base_mask(1)])
home=R.shortest_fields(after,j.broadcast_to(cages,(1,2,13,15)))
fn=jax.jit(F.native_frontier)
shape=jax.eval_shape(fn,st,after,info,aux,seats,home,j.zeros(1,j.int32));check('actual_native_schema_trace',all(x.shape==(1,) for x in shape.values()))
proof=fn(st,after,info,aux,seats,home,j.zeros(1,j.int32));jax.block_until_ready(proof)
print('proof',jax.tree.map(lambda x:x.tolist(),proof),'start',st.pos.tolist(),'after',after.pos.tolist(),flush=True)
check('native_selected_model_binds_current_env',native_safety().env is env)
check('native_actual_own_moved',proof['moved'][0]);check('native_safe_positive_crossing',proof['c_eligible'][0]&proof['strict_crossing'][0]&proof['selected_survivable'][0]&proof['post_safe'][0])
p,c,h=F.event_gate(j.zeros(1,j.int32),**proof,bad=j.array([False]),terminal=j.array([False]),concession=j.array([False]));check('native_positive_event03',j.allclose(p,.03))
# Neither an unchanged own position nor enemy-only movement can create the event.
no_move=after._replace(core=after.core._replace(pos=st.pos))
z=fn(st,no_move,info,aux,seats,home,j.zeros(1,j.int32));check('enemy_only_or_unchanged_own_no_c',~z['c_eligible'][0])
seen=fn(st,after,info,aux,seats,home,j.array([4096],j.int32));check('once_flag_short_circuits_native_event',~seen['c_eligible'][0])
# Pre-existing own/opponent fire lanes must fail the post/swept certificate (even if mask is legal).
hazard=st._replace(core=st.core._replace(fuse=st.fuse.at[0,6,6].set(20),owner=st.core.owner.at[0,6,6].set(1),bomb_blast=st.core.bomb_blast.at[0,6,6].set(6)))
hazard_after=after._replace(core=after.core._replace(fuse=after.fuse.at[0,6,6].set(19),owner=after.core.owner.at[0,6,6].set(1),bomb_blast=after.core.bomb_blast.at[0,6,6].set(6)))
z=fn(hazard,hazard_after,info,aux,seats,home,j.zeros(1,j.int32));check('legal_but_crosses_enemy_fire_rejected',~z['post_safe'][0])
# Unreachable/empty board and no live enemy are not target evidence.
dead=st._replace(core=st.core._replace(alive=st.alive.at[0,1].set(False)))
z=fn(dead,after,info,aux,seats,home,j.zeros(1,j.int32));check('enemy_dead_no_frontier',~z['c_eligible'][0])
# Native movement model, RNG and original state remain immutable under evidence computation.
after2,_,_=step(st,actions,jax.random.split(jax.random.PRNGKey(43),1));check('observer_does_not_modify_native_transition',all(bool(j.array_equal(x,y)) for x,y in zip(jax.tree.leaves(after),jax.tree.leaves(after2))))
atomic_json(OUT/'NATIVE_INTEGRATION.json',dict(status='PASS',tests=tests,cpu_backend=jax.default_backend(),unix=time.time(),sources={f:sha(f) for f in ('scripts/cx_safe_frontier_v1.py','scripts/cx_fountain_continuation_v2.py','scripts/cx_ftn2_reward_v1.py')},proof=jax.tree.map(lambda x:x.tolist(),proof)))
print('PASS',len(tests),'real native CPU tests',flush=True)
