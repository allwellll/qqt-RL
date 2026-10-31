"""ITER2 D3: complete physical JAX Hunter games, no optimizer/reward ledger."""
import os,json,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha,lock
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1/iter2'


def main():
    manifest=json.loads((BASE/'D123_MANIFEST.json').read_text());out=BASE/'D3';out.mkdir(exist_ok=True)
    guard=lock(out/'worker.lock')
    if (out/'COMPLETE.json').exists():return
    amend=BASE/'D123_EXECUTION_AMENDMENT.json';runtime=json.loads(amend.read_text()) if amend.exists() else {}
    for f,h in manifest['source_sha256'].items():assert sha(ROOT/f)==runtime.get('source_sha256',{}).get(f,h),f
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    import jax,jax.numpy as j,numpy as np
    from scripts import carrier_mechanism_engine as E
    from scripts.bomb_threat_runtime import install
    from scripts.checkpoint_policy_artifact import load_actor
    E.configure();install();assert E.bun_jax_bots.TIER_NAMES[3]=='hunter'
    spec=next(s for s in manifest['models'] if s['name']==manifest['D3_model'])
    assert sha(ROOT/spec['checkpoint']/'policy.pt')==spec['policy_sha256']
    params=jax.tree.map(j.asarray,load_actor(ROOT/spec['checkpoint']))
    spawns=j.asarray(json.loads((ROOT/manifest['spawn_manifest']).read_text())['spawn_cells'],j.float32)+.5
    started=time.time();results=[];files={};n=8
    def initialize(ids):
        envkeys=jax.vmap(lambda i:jax.random.PRNGKey(manifest['seed']+i))(ids)
        st,_=E.initial(envkeys,j.array(0,j.int32))
        pos=spawns[ids]
        st=st._replace(core=st.core._replace(pos=pos),bun_spawn_pos=pos)
        keys=jax.vmap(lambda k:jax.random.fold_in(k,0x515154))(envkeys)
        return st,keys
    initialize=jax.jit(initialize)
    def chunk(params,st,keys,ended,seats):
        def body(c,_):
            st,keys,ended=c
            nk,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,3))(keys),0,1)
            allobs=E.train.both_perspectives(st);allgv=E.train.both_states(st)
            idx=j.arange(n)+seats*n
            mv,ab,_,_=E.train.net_forward(params,'transformer',allobs[idx],allgv[idx])
            mm,bm=jax.vmap(E.env.legal_mask)(st)
            rows=j.arange(n);own=j.stack([j.argmax(j.where(mm[rows,seats],mv,-1e9),-1),
                j.argmax(j.where(bm[rows,seats],ab,-1e9),-1)],-1)
            def opponent(s,seat,m,b,k):
                return E.bun_jax_bots.rule_bot_actions(jax.tree.map(lambda x:x[None],s),seat[None],m[None],b[None],k,j.array([3]))[0]
            enemy=jax.vmap(opponent)(st,1-seats,mm[rows,1-seats],bm[rows,1-seats],ko)
            ac=j.zeros((n,2,2),j.int32).at[rows,seats].set(own).at[rows,1-seats].set(enemy)
            after,done,info=jax.vmap(lambda s,a,k:E.env.step(s,a,k,auto_reset=False,return_info=True))(st,ac,ke)
            alive=~ended
            metrics=dict(live=alive,done=done&alive,winner=info['winner'],tick=after.t,
                actions=ac,placed=info['bomb_placed'],causal_kills=info['causal_kill'],deaths=info['death'],
                stored=after.bun_stored)
            return (E.select_tree(ended,st,after),j.where(ended[:,None],keys,nk),ended|done),metrics
        return jax.lax.scan(body,(st,keys,ended),None,length=128)
    scan=jax.jit(chunk)
    for start in range(0,64,n):
        p=out/f'games{start:03d}.json'
        if p.exists():
            old=json.loads(p.read_text());assert old['policy_sha256']==spec['policy_sha256']
            for f,h in old['files'].items():assert sha(ROOT/f)==h
            results+=old['episodes'];files[str(p.relative_to(ROOT))]=sha(p);continue
        ids=j.arange(start,start+n,dtype=j.int32);seats=ids%2
        st,keys=initialize(ids);jax.block_until_ready(st)
        assert np.all(np.asarray(st.bun_carried)<0) and np.all(np.asarray(st.t)==0)
        assert np.all(np.asarray(st.lesson)==E.env.LESSON_FULL)
        carry=(st,keys,j.zeros(n,bool));traces=[];tracefiles={}
        for offset in range(0,2432,128):
            tick=time.monotonic();carry,data=scan(params,*carry,seats);jax.block_until_ready(carry)
            path=out/f'games{start:03d}_ticks{offset:04d}.json'
            atomic_json(path,jax.tree.map(lambda x:np.asarray(x).tolist(),data));tracefiles[str(path.relative_to(ROOT))]=sha(path)
            traces.append(jax.tree.map(np.asarray,data))
            atomic_json(out/'PROGRESS.json',dict(shard=start,ticks=offset+128,completed_games=len(results),
                scan_seconds=time.monotonic()-tick,elapsed_seconds=time.time()-started,unix=time.time()))
            if np.all(np.asarray(carry[2])):break
        alltrace={k:np.concatenate([t[k] for t in traces]) for k in traces[0]};rows=[]
        for i in range(n):
            v=alltrace['live'][:,i];terminal=np.flatnonzero(alltrace['done'][:,i]);assert len(terminal)==1
            end=terminal[0];seat=int(seats[i])
            rows.append(dict(seed=manifest['seed']+start+i,seat=seat,ticks=int(v.sum()),done=True,
                winner=int(alltrace['winner'][end,i]),placed=alltrace['placed'][v,i].sum(0).tolist(),
                requests=(alltrace['actions'][v,i,:,1]==1).sum(0).tolist(),
                causal_kills=alltrace['causal_kills'][v,i].sum(0).tolist(),deaths=alltrace['deaths'][v,i].sum(0).tolist(),
                final_stock=alltrace['stored'][end,i].tolist()))
        atomic_json(p,dict(episodes=rows,policy_sha256=spec['policy_sha256'],files=tracefiles))
        results+=rows;files[str(p.relative_to(ROOT))]=sha(p)
    assert len(results)==64 and {r['seed'] for r in results}==set(range(manifest['seed'],manifest['seed']+64))
    wins=sum(r['winner']==r['seat'] for r in results);losses=sum(r['winner'] not in (-1,r['seat']) for r in results)
    summary=dict(games=64,win=wins,draw=64-wins-losses,loss=losses,win_minus_loss=wins-losses,
        causal_kills=sum(r['causal_kills'][r['seat']] for r in results),deaths=sum(r['deaths'][r['seat']] for r in results),
        bombs_per_game=sum(r['placed'][r['seat']] for r in results)/64,zero_bomb_games=sum(r['placed'][r['seat']]==0 for r in results))
    atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',model=spec['name'],policy_sha256=spec['policy_sha256'],
        episodes=results,metrics=summary,files=files,elapsed_seconds=time.time()-started,
        opponent='jax_bun_jax_bots hunter tier index3 (same implementation called rule3 elsewhere)',
        seed_and_spawn_pairs_match_JS=True,seat_alternates=True,action_mode='greedy',teacher=False,
        limitations=['JAX and JS PRNG algorithms differ','not a same-action dynamics parity test',
            'different environment/opponent implementations: differences cannot identify weaker opponent alone',
            'physical first terminal only; no reward/ledger tail or critic accuracy measurement'],unix=time.time()))


if __name__=='__main__':main()
