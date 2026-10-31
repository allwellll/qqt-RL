"""CPU-only natural JAX categorical policy probe, no optimizer or training owner."""
import json,os,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
B=Path('runs/bomb_threat_v1/jitter_followup_u800')
def main():
    guard=lock(B/'NATIVE.lock');cfg=json.loads((B/'PROTOCOL.json').read_text());out=B/'native';out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    import jax,jax.numpy as j,numpy as np
    from scripts import carrier_mechanism_engine as E
    from scripts.bomb_threat_runtime import install
    from scripts.checkpoint_policy_artifact import load_actor
    E.configure();install();assert jax.default_backend()=='cpu'
    cp=Path(cfg['model']['checkpoint']);assert sha(cp/'policy.pt')==cfg['model']['policy_sha256'];params=jax.tree.map(j.asarray,load_actor(cp))
    n=cfg['native_probe']['games'];sp=j.asarray(json.loads(Path(cfg['spawn_manifest']).read_text())['spawn_cells'][:n],j.float32)+.5
    keys=jax.vmap(jax.random.PRNGKey)(j.asarray(cfg['native_probe']['seeds'],j.uint32));st,_=jax.jit(E.initial)(keys,j.asarray(0));st=st._replace(core=st.core._replace(pos=sp),bun_spawn_pos=sp)
    assert np.all(np.asarray(st.bun_carried)<0);assert np.all(np.asarray(st.lesson)==E.env.LESSON_FULL)
    def chunk(params,st,keys,ended):
        def body(c,_):
            st,keys,ended=c;nk,ka,kb,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,5))(keys),0,1)
            o,g,mm,bm=E.inputs(st);mv,ab,*_=E.train.net_forward(params,'transformer',o,g)
            own=j.stack([jax.vmap(jax.random.categorical)(ka,j.where(mm,mv,-1e9)),jax.vmap(jax.random.categorical)(kb,j.where(bm,ab,-1e9))],-1)
            allmm,allbm=jax.vmap(E.env.legal_mask)(st)
            def enemy(s,m,b,k):return E.bun_jax_bots.rule_bot_actions(jax.tree.map(lambda x:x[None],s),j.array([1]),m[None],b[None],k,j.array([3]))[0]
            foe=jax.vmap(enemy)(st,allmm[:,1],allbm[:,1],ko);ac=j.stack([own,foe],1)
            after,done,info=jax.vmap(lambda s,a,k:E.env.step(s,a,k,auto_reset=False,return_info=True))(st,ac,ke)
            data=dict(tick=st.t,live=~ended,controlled=st.alive[:,0]&(st.trapped[:,0]==0),enemy_controlled=st.alive[:,1]&(st.trapped[:,1]==0),move=own[:,0],ability=own[:,1],enemy_move=foe[:,0],pos=st.pos,done=done&~ended,winner=info['winner'],carry=st.bun_carried[:,0],placed=info['bomb_placed'],deaths=info['death'])
            return (E.select_tree(ended,st,after),j.where(ended[:,None],keys,nk),ended|done),data
        return jax.lax.scan(body,(st,keys,ended),None,length=128)
    scan=jax.jit(chunk);carry=(st,keys,j.zeros(n,bool));traces=[];started=time.time()
    atomic_json(out/'STATUS.json',dict(status='COMPILING_OR_SCANNING',cpu_only=True,games=n,unix=time.time()))
    for offset in range(0,2432,128):
        p=out/f'ticks{offset:04d}.npz';tick=time.monotonic();carry,data=scan(params,*carry);jax.block_until_ready(carry);host=jax.tree.map(np.asarray,data);np.savez_compressed(p,**host);traces.append(host)
        atomic_json(out/'STATUS.json',dict(status='RUNNING',ticks=offset+128,scan_seconds=time.monotonic()-tick,elapsed_seconds=time.time()-started,unix=time.time()))
        if np.all(np.asarray(carry[2])):break
    merged={k:np.concatenate([d[k] for d in traces]) for k in traces[0]};episodes=[]
    for i,seed in enumerate(cfg['native_probe']['seeds']):
        live=merged['live'][:,i];controlled=merged['controlled'][:,i]&live;actions=merged['move'][:,i];events=np.flatnonzero(controlled[1:]&controlled[:-1]&(((actions[:-1]==2)&(actions[1:]==3))|((actions[:-1]==3)&(actions[1:]==2))))+1;terminal=np.flatnonzero(merged['done'][:,i]);assert len(terminal)==1
        episodes.append(dict(seed=seed,seat=0,ticks=int(live.sum()),done=True,winner=int(merged['winner'][terminal[0],i]),move_ticks=int(np.sum(controlled&(actions<4))),lr_reversals=len(events),reversal_intervals=np.diff(events).tolist(),placed=merged['placed'][live,i].sum(0).tolist(),deaths=merged['deaths'][live,i].sum(0).tolist()))
    atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',episodes=episodes,model_sha256=cfg['model']['policy_sha256'],opponent='native JAX Hunter tier3',mode='categorical',seat=0,natural=True,training_collector_history=False,limitation='CPU same training-native policy+physics natural probe; no Bank/curriculum/mixed opponents/optimizer; JS vs native+opponent+RNG confounded',files={str(p):sha(p) for p in out.glob('ticks*.npz')},elapsed_seconds=time.time()-started,unix=time.time()))
if __name__=='__main__':main()
