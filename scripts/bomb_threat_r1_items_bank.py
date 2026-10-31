"""Collect both seats from fresh native-item episodes and real stock windows."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from scripts.bomb_threat_r1_items_runtime import install
install()
import json,pickle,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from scripts import combat_item_engine as T,carrier_mechanism_engine as E
from scripts.population_run_state import atomic_json,sha,lock
from scripts.terminal_checkpoint_restore import digest
from scripts.bomb_threat_runtime import install as stock_window
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030/items_v2')


def save(path,data):
    with path.with_suffix('.tmp').open('wb') as f:pickle.dump(data,f,protocol=5);f.flush();os.fsync(f.fileno())
    path.with_suffix('.tmp').replace(path)


def main():
    guard=lock(OUT/'BANK.lock');began=time.time();T.configure();stock_window()
    if (OUT/'raw_bank.pkl').exists():
        verify(pickle.load((OUT/'raw_bank.pkl').open('rb')),began);return
    n=256;keys=jax.random.split(jax.random.PRNGKey(2026183091),n);seats=j.arange(n)%2;rows=j.arange(n)
    st,ax=E.initial(keys,0);ax=dict(ax,tactics=T.events.state(n))
    initial=jax.device_get((st,ax,keys));ended=j.zeros(n,bool);got=np.zeros(n,bool);records=[];actions=[];envkeys=[]
    from jax_bomb import bun_jax_bots as bots
    from scripts.je10_route_oracle import labels
    from scripts.bomb_threat_seat5050_collector import inputs,joint_actions
    def step(st,ax,keys,ended):
        nk,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,3))(keys),0,1)
        mm,bm=jax.vmap(E.env.legal_mask)(st)
        a0=bots.rule_bot_actions(st,j.zeros(n,j.int32),mm[:,0],bm[:,0],ko[0],j.full(n,4))
        a1=bots.rule_bot_actions(st,j.ones(n,j.int32),mm[:,1],bm[:,1],ko[0],j.full(n,4))
        own=j.where((seats==0)[:,None],a0,a1);enemy=j.where((seats==0)[:,None],a1,a0)
        o,g,m,b=inputs(st,seats);target=labels(st,ax,m,seats,o[:,5])[:,:5]
        window=(~st.alive[rows,1-seats]|(st.trapped[rows,1-seats]>0))&(st.bun_carried[rows,seats]<0)
        own=own.at[:,0].set(j.where(window&(target.sum(-1)>0),target.argmax(-1),own[:,0]))
        own=own.at[:,1].set(j.where(window,0,own[:,1]));act=joint_actions(own,enemy,seats)
        after,na,r,d,info,parts=T.transition(st,ax,act,ke,'control')
        take=(st.bun_carried[rows,seats]<0)&(after.bun_carried[rows,seats]==1-seats)&info['steal'][rows,seats]&window&~ended
        return after,na,nk,ended|d,take,act,ke
    step=jax.jit(step)
    for tick in range(2400):
        before,preax,prek=st,ax,keys
        st,ax,keys,ended,take,ac,ek=step(st,ax,keys,ended)
        take,ac,ek=jax.device_get((take,ac,ek));actions.append(ac);envkeys.append(ek)
        for i in np.flatnonzero(take&~got):
            get=lambda tr:jax.tree.map(lambda a:np.asarray(a[i:i+1]),jax.device_get(tr))
            seat=int(seats[i]);old=get(before);after=get(st)
            assert int(old.bun_carried[0,seat])<0 and int(after.bun_carried[0,seat])==1-seat
            assert int(after.bun_stored[0,1-seat,1-seat])==int(old.bun_stored[0,1-seat,1-seat])-1
            records.append(dict(record=dict(source_episode=f'fresh-native-items-{i}',source_seed=2026183091,source_row=int(i),seat=seat,map_id=806,split='train',event_tick=int(after.t[0]),
                before_carried=-1,after_carried=1-seat,actual_env_step=True,enemy_alive=bool(old.alive[0,1-seat]),enemy_trapped=int(old.trapped[0,1-seat]),action_source='real rule/route continuation, not autonomous capability'),
                states=after,aux=get(ax),keys=get(keys),before=old,preaux=get(preax),initial=get(initial),
                acquisition_actions=ac[i:i+1],acquisition_envkeys=ek[i:i+1],prefix_actions=np.asarray(actions)[:,i:i+1],prefix_envkeys=np.asarray(envkeys)[:,i:i+1]))
        got|=take
        counts=[sum(x['record']['seat']==s for x in records) for s in (0,1)]
        if tick%16==15:atomic_json(OUT/'BANK_PROGRESS.json',dict(status='COLLECTING',ticks=tick+1,counts=counts,source_episodes=n,wall_seconds=time.time()-began))
        if min(counts)>=16:break
    assert min(counts)>0,'no genuine both-seat acquisition coverage'
    save(OUT/'raw_bank.pkl',records)
    verify(records,began,n)


def verify(records,began,n=256):
    counts=[sum(x['record']['seat']==s for x in records) for s in (0,1)]
    manifest=dict(status='COLLECTED_AWAITING_PREFIX_PARITY',counts=counts,source_episodes=n,map_id=806,precarried_initialization=False,teleportation=False,raw_sha256=sha(OUT/'raw_bank.pkl'),wall_seconds=time.time()-began)
    atomic_json(OUT/'BANK_COLLECTED.json',manifest)
    # Use identical batches for prefix replay; record all row differences.
    from scripts.carrier_mechanism_analysis import merged
    lengths=j.array([len(x['prefix_actions']) for x in records]);h=int(lengths.max());k=len(records)
    initial=jax.tree.map(lambda *v:np.concatenate(v),*[x['initial'] for x in records]);ac=np.zeros((h,k,2,2),np.int32);ek=np.zeros((h,k,2),np.uint32)
    for i,x in enumerate(records):
        l=len(x['prefix_actions']);ac[:l,i]=x['prefix_actions'][:,0];ek[:l,i]=x['prefix_envkeys'][:,0]
    def replay(st,ax):
        def body(c,data):
            a,key,t=data;ns,na,*_=T.transition(c[0],c[1],a,key,'control')
            return (E.select_tree(t<lengths,ns,c[0]),E.select_tree(t<lengths,na,c[1])),None
        return jax.lax.scan(body,(st,ax),(j.asarray(ac),j.asarray(ek),j.arange(h)))[0]
    ss,aa=jax.jit(replay)(initial[0],initial[1]);jax.block_until_ready(ss.t)
    good=[];differences=[]
    for i,x in enumerate(records):
        actual=jax.tree.map(lambda a:np.asarray(a[i:i+1]),(ss,aa));expected=(x['states'],x['aux'])
        ds=[]
        for (path,a),(_,b) in zip(jax.tree_util.tree_flatten_with_path(actual)[0],jax.tree_util.tree_flatten_with_path(expected)[0]):
            if not np.array_equal(a,b):ds.append(dict(field=jax.tree_util.keystr(path),max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float))))))
        if ds:differences.append(dict(row=i,differences=ds))
        else:good.append(x)
    counts=[sum(x['record']['seat']==s for x in good) for s in (0,1)];assert min(counts)>0,'no exact prefix bank both seats'
    manifest.update(status='PASS',counts=counts,excluded_rows=differences,selection='exact prefix env/aux parity only, no outcome selection',limitations=['single map806','rule/route source, not autonomous policy','real attribute baselines, no pre-source progress paid'])
    for seat in (0,1):
        path=OUT/f'p{seat}_bank.pkl';save(path,[x for x in good if x['record']['seat']==seat]);manifest[f'p{seat}_file']=str(path);manifest[f'p{seat}_sha256']=sha(path)
    manifest.update(wall_seconds=time.time()-began,unix=time.time());atomic_json(OUT/'BANK_FROZEN.json',manifest)

if __name__=='__main__':main()
