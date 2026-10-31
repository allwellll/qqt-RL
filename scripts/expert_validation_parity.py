"""C: fixed-state/action JAX-vs-production-JS parity witness.

The fixture deliberately has no bombs, item releases, deaths, or random drops;
its RNG stream is therefore not consumed.  Eventful physics remains outside
this witness and is reported as a limitation rather than silently conflated.
"""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from jax_bomb import je07_bun_env as env
from scripts import carrier_mechanism_engine as E,combat_item_engine as T,je10_ops as ops
from jax_bomb import jax_train as train

R=Path(__file__).resolve().parents[1]
B=R/'runs/expert_validation_20261025'
SEED=2026102860; TICKS=48

def h(x): return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def prepare_state():
    T.configure()
    init=jax.jit(lambda k:jax.tree.map(lambda x:x[0],env.init_batch(k,1,jax.nn.one_hot(0,16))))
    st=init(jax.random.PRNGKey(SEED))
    core=st.core._replace(pos=jnp.asarray([[4.5,2.5],[4.5,12.5]],jnp.float32),alive=jnp.ones((2,),bool),hp=jnp.ones((2,),jnp.int32),invuln=jnp.zeros((2,),jnp.int32),fuse=jnp.zeros_like(st.core.fuse),owner=jnp.full_like(st.core.owner,-1),bomb_blast=jnp.zeros_like(st.core.bomb_blast),blast_linger=jnp.zeros_like(st.core.blast_linger),brick_linger=jnp.zeros_like(st.core.brick_linger),bombs_cap=jnp.asarray([2.,2.]),blast_cap=jnp.asarray([2.,2.]),spd_g=jnp.asarray([1.3,1.3]))
    st=st._replace(core=core,bun_carried=jnp.full((2,),-1,jnp.int8),bun_stored=jnp.asarray([[1,0],[0,1]],jnp.int8),bun_loose=jnp.zeros_like(st.bun_loose),bun_respawn=jnp.zeros((2,),jnp.int16),bun_spawn_pos=core.pos,held_item=jnp.zeros((2,),jnp.int8),field_item=jnp.zeros_like(st.field_item),field_owner=jnp.full_like(st.field_owner,-1),field_armed=jnp.zeros_like(st.field_armed),trapped=jnp.zeros((2,),jnp.int16),item_slots=jnp.zeros_like(st.item_slots),item_counts=jnp.zeros_like(st.item_counts),scene=jnp.asarray(8,jnp.int8),lesson=jnp.asarray(env.LESSON_FULL,jnp.int8))
    return st
def action(t): return jnp.asarray([[3 if t%8<4 else 2,0],[2 if t%8<4 else 3,0]],jnp.int32)
def main():
    st=prepare_state();print('C initialized',flush=True)
    acts=np.asarray([action(t) for t in range(TICKS)],np.int32)
    key=jax.random.PRNGKey(SEED^0xabcdef);keys=jax.random.split(key,TICKS)
    @jax.jit
    def rollout(initial,action_rows,random_keys):
        def body(current,pair):
            act,rng=pair;after,done,info=env.step(current,act,rng,auto_reset=False,return_info=True)
            return after,after
        _,tail=jax.lax.scan(body,initial,(action_rows,random_keys))
        states=jax.tree.map(lambda first,rest:jnp.concatenate([first[None],rest],axis=0),initial,tail)
        obs=train.both_perspectives(states);mm,bm=jax.vmap(env.legal_mask)(states)
        return states,obs,mm,bm
    states,obs,mm,bm=rollout(st,jnp.asarray(acts),keys);jax.block_until_ready(obs);print('C JAX rollout complete',flush=True)
    leaves=jax.device_get(states);obs=np.asarray(obs).reshape(2,TICKS+1,24,13,15).transpose(1,0,2,3,4);mm=np.asarray(mm);bm=np.asarray(bm)
    trace=[]
    for i in range(TICKS+1):
        get=lambda name:np.asarray(getattr(leaves,name)[i])
        trace.append({'t':int(get('t')),'pos':get('pos').reshape(-1).tolist(),'alive':get('alive').tolist(),'trapped':get('trapped').tolist(),'bun_carried':get('bun_carried').tolist(),'bun_stored':get('bun_stored').tolist(),'brick_hash':h(get('brick').reshape(-1).astype(int).tolist()),'wall_hash':h(get('wall').reshape(-1).astype(int).tolist()),'crate_hash':h(np.asarray(leaves.core.crate[i]).reshape(-1).astype(int).tolist()),'fuse_hash':h(get('fuse').reshape(-1).astype(int).tolist()),'field_item_hash':h(get('field_item').reshape(-1).astype(int).tolist()),'held_item':get('held_item').tolist(),'move_status':get('move_status').tolist(),'move_mask':mm[i].astype(int).tolist(),'ability_mask':bm[i].astype(int).tolist(),'observation':obs[i].reshape(2,-1).tolist()})
    print('C JAX trace serialized',flush=True)
    jsout=B/'C_js_trace.json';subprocess.run(['node','scripts/expert_validation_js_parity.js',str(jsout)],cwd=R,check=True)
    js=json.loads(jsout.read_text());fields=['pos','alive','trapped','bun_carried','bun_stored','brick_hash','wall_hash','crate_hash','fuse_hash','field_item_hash','held_item','move_status','move_mask','ability_mask'];first_by_field={};max_obs_error=0.0;max_position_error=0.0;max_velocity_proxy_error=0.0
    for i,(a,b) in enumerate(zip(trace,js['trace'])):
        for f in fields:
            if f=='pos':
                error=float(np.max(np.abs(np.asarray(a[f],float)-np.asarray(b[f],float))));max_position_error=max(max_position_error,error);ok=error<=2e-5
            else: error=None;ok=a[f]==b[f]
            if not ok and f not in first_by_field:first_by_field[f]={'tick':i,'max_abs_error':error,'jax':a[f] if f!='pos' else None,'js':b[f] if f!='pos' else None}
        obs_error=float(np.max(np.abs(np.asarray(a['observation'],np.float32)-np.asarray(b['observation'],np.float32))))
        max_obs_error=max(max_obs_error,obs_error)
        if obs_error>2e-5 and 'observation' not in first_by_field:first_by_field['observation']={'tick':i,'max_abs_error':obs_error}
        if i and a['pos'] and b['pos']:
            jax_delta=np.asarray(a['pos'])-np.asarray(trace[i-1]['pos']);js_delta=np.asarray(b['pos'])-np.asarray(js['trace'][i-1]['pos']);max_velocity_proxy_error=max(max_velocity_proxy_error,float(np.max(np.abs(jax_delta-js_delta))))
    report={'schema':'qqt.expert-validation-dynamics-parity/v1','seed':SEED,'ticks':TICKS,'actions':acts.tolist(),'engines':['JAX je07_bun_env native transition','production JS web/sim.js'],'rng':'explicit JAX keys per step; no stochastic event branch consumed','fields':fields+['observation','position_delta_proxy'],'tolerances':{'position':2e-5,'observation':2e-5,'boolean_integer_state':'exact'},'jax_trace':trace,'js_trace_file':'runs/expert_validation_20261025/C_js_trace.json','first_deviation_by_field':first_by_field,'deviation_count':len(first_by_field),'max_position_error':max_position_error,'max_velocity_proxy_error':max_velocity_proxy_error,'max_observation_error':max_obs_error,'limitations':['No bombs, item releases, deaths or random drops in this witness; eventful RNG/physics requires a separately seeded event fixture.','Velocity is not a stored field in either engine; position deltas are the observable motion proxy.'],'source_sha256':{'scripts/expert_validation_parity.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scripts/expert_validation_js_parity.js':hashlib.sha256((R/'scripts/expert_validation_js_parity.js').read_bytes()).hexdigest()}}
    (B/'C_dynamics_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'first_deviation_by_field':report['first_deviation_by_field'],'deviation_count':report['deviation_count'],'max_position_error':report['max_position_error'],'max_observation_error':report['max_observation_error']}),flush=True)
if __name__=='__main__': main()
