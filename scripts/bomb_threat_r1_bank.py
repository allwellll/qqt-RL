"""R1 P1 Bank: actual native acquisition after an empty-hand saved state.

No identity relabeling, artificial carried objects or relocated players. The
source is an atomic training state; actions are diagnostic rule/route actions,
not autonomous student evidence. Independent source rows are kept explicitly.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,pickle,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
from scripts.population_run_state import atomic_json,sha,lock
from scripts import combat_item_engine as T,carrier_mechanism_engine as E
from scripts.bomb_threat_runtime import install
from scripts.terminal_checkpoint_restore import digest
OUT=Path('runs/bomb_threat_v1/iter2/r1_attack_20261030')


def main():
    guard=lock(OUT/'BANK.lock');began=time.time();T.configure();T.C=None;install()
    if (OUT/'p1_bank.pkl').exists():
        verify(pickle.load((OUT/'p1_bank.pkl').open('rb')),began);return
    src=json.loads((OUT/'SOURCE_AUTHORIZATION.json').read_text())['source'];cp=Path(src['checkpoint'])
    assert sha(cp/'state.pkl')==src['state_sha256']
    p=pickle.load((cp/'state.pkl').open('rb'))
    st,ax,keys=jax.tree.map(lambda a:np.asarray(a)[0],(p['states'],p['aux'],p['keys']))
    select=(ax['combat_mix']['learner_seat']==1)&(ax['provenance_source']<0)&~ax['terminal']&(st.bun_carried[:,1]<0)&st.alive[:,1]&(st.trapped[:,1]==0)
    ids=np.flatnonzero(select)[:128];assert len(ids)>=16
    st,ax,keys=jax.tree.map(lambda a:j.asarray(a[ids]),(st,ax,keys))
    ax={k:v for k,v in ax.items() if k not in ('combat_mix','a3sr')}
    initial=jax.device_get((st,ax,keys));n=len(ids);ended=j.zeros(n,bool);got=np.zeros(n,bool);records=[];actions=[];envkeys=[]
    from jax_bomb import bun_jax_bots as bots
    from scripts.je10_route_oracle import labels
    from scripts.bomb_threat_seat5050_collector import inputs
    def step(st,ax,keys,ended):
        nk,ko,ke=j.swapaxes(jax.vmap(lambda k:jax.random.split(k,3))(keys),0,1)
        mm,bm=jax.vmap(E.env.legal_mask)(st)
        acts=[]
        for seat in (0,1):
            a=bots.rule_bot_actions(st,j.full(n,seat),mm[:,seat],bm[:,seat],ko[0],j.full(n,4))
            if seat==1:
                o,g,m,b=inputs(st,j.ones(n,j.int32));target=labels(st,ax,m,j.ones(n,j.int32),o[:,5])[:,:5]
                native=(~st.alive[:,0]|(st.trapped[:,0]>0))&(st.bun_carried[:,1]<0)
                a=a.at[:,0].set(j.where(native&(target.sum(-1)>0),target.argmax(-1),a[:,0]))
                a=a.at[:,1].set(j.where(native,0,a[:,1]))
            acts.append(a)
        act=j.stack(acts,1)
        after,na,r,d,info,parts=T.transition(st,ax,act,ke,'control')
        take=(st.bun_carried[:,1]<0)&(after.bun_carried[:,1]==0)&info['steal'][:,1]&(~st.alive[:,0]|(st.trapped[:,0]>0))&st.alive[:,1]&after.alive[:,1]&~ended
        return after,na,nk,ended|d,take,act,ke,info
    step=jax.jit(step)
    for t in range(1024):
        before,preax,prek=st,ax,keys
        st,ax,keys,ended,take,ac,ek,info=step(st,ax,keys,ended)
        take,ac,ek=jax.device_get((take,ac,ek));actions.append(ac);envkeys.append(ek)
        for i in np.flatnonzero(np.asarray(take)&~got):
            get=lambda tr:jax.tree.map(lambda a:np.asarray(a[i:i+1]),jax.device_get(tr))
            after=get(st);old=get(before);iaux=get(ax)
            assert int(old.bun_carried[0,1])<0 and int(after.bun_carried[0,1])==0
            assert int(after.bun_stored[0,0,0])==int(old.bun_stored[0,0,0])-1
            record=dict(source_checkpoint=src['checkpoint'],source_state_sha256=src['state_sha256'],source_row=int(ids[i]),seat=1,
                source_episode=f"r1-atomic-row-{int(ids[i])}",split='train',map_id=806,event_tick=int(after.t[0]),ticks_from_source=t+1,
                actual_env_step=True,stock_steal=True,enemy_alive=bool(old.alive[0,0]),enemy_trapped=int(old.trapped[0,0]),
                before_carried=-1,after_carried=0,pre_hash=digest(old),post_hash=digest(after),aux_hash=digest(iaux),
                action_source='native rule bot plus legal route oracle in actual native window; not student autonomy')
            records.append(dict(record=record,states=after,aux=iaux,keys=get(keys),before=old,preaux=get(preax),prekey=get(prek),
                acquisition_actions=ac[i:i+1],acquisition_envkeys=ek[i:i+1],
                initial=get(initial),prefix_actions=np.asarray(actions)[:,i:i+1],prefix_envkeys=np.asarray(envkeys)[:,i:i+1]))
        got|=np.asarray(take)
        if (t+1)%16==0:
            atomic_json(OUT/'BANK_PROGRESS.json',dict(status='COLLECTING',ticks=t+1,source_rows=n,acquisitions=int(got.sum()),wall_seconds=time.time()-began))
        if got.sum()>=16:break
    assert len(records)>=1,'P1 genuine Bank coverage failed; cannot fabricate fallback'
    path=OUT/'p1_bank.pkl'
    with path.open('wb') as f:pickle.dump(records,f,protocol=5)
    verify(records,began)


def verify(records,began):
    path=OUT/'p1_bank.pkl';n=len(records);lengths=j.array([len(x['prefix_actions']) for x in records]);h=int(lengths.max())
    initial=jax.tree.map(lambda *a:np.concatenate(a),*[x['initial'] for x in records])
    act=np.zeros((h,n,2,2),np.int32);ek=np.zeros((h,n,2),np.uint32)
    for i,x in enumerate(records):
        l=len(x['prefix_actions']);act[:l,i]=x['prefix_actions'][:,0];ek[:l,i]=x['prefix_envkeys'][:,0]
    def replay(st,ax,act,ek):
        def body(c,data):
            ac,k,t=data;ns,na,*_=T.transition(c[0],c[1],ac,k,'control')
            return (E.select_tree(t<lengths,ns,c[0]),E.select_tree(t<lengths,na,c[1])),None
        return jax.lax.scan(body,(st,ax),(act,ek,j.arange(h)))[0]
    cache=OUT/'p1_replay.pkl'
    if cache.exists():ss,aa=pickle.load(cache.open('rb'))
    else:
        ss,aa=jax.jit(replay)(initial[0],initial[1],j.asarray(act),j.asarray(ek));jax.block_until_ready(ss.t)
        with cache.open('wb') as f:pickle.dump(jax.device_get((ss,aa)),f,protocol=5)
    differences=[]
    for i,x in enumerate(records):
        st,ax=jax.tree.map(lambda a:np.asarray(a[i:i+1]),(ss,aa))
        for family,one,two in [('env',st,x['states']),('aux',ax,x['aux'])]:
            for (leafpath,a),(_,b) in zip(jax.tree_util.tree_flatten_with_path(one)[0],jax.tree_util.tree_flatten_with_path(two)[0]):
                if not np.array_equal(a,b):differences.append(dict(row=i,family=family,field=jax.tree_util.keystr(leafpath),max_abs=float(np.max(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)))),dtype=str(a.dtype)))
    atomic_json(OUT/'BANK_REPLAY_DIFFERENCES.json',dict(status='PASS' if not differences else 'DIFFERENCES',differences=differences))

    # Exclude numerical replay failures rather than rewriting their full state
    # or claiming approximate equality is exact. Selection uses parity only.
    excluded=sorted({d['row'] for d in differences})
    for d in differences:
        assert d['family']=='aux' and d['field'] in ("['tactics']['banana_frontier']","['tactics']['glue_frontier']") and d['max_abs']<=5e-6,d
    exact_records=[x for i,x in enumerate(records) if i not in excluded]
    exactpath=OUT/'p1_bank_exact.pkl'
    with exactpath.open('wb') as f:pickle.dump(exact_records,f,protocol=5)
    checked=[]
    for i,x in enumerate(records):
        if i in excluded:continue
        st,ax=jax.tree.map(lambda a:a[i:i+1],(ss,aa))
        assert digest(st)==digest(x['states']) and digest(ax)==digest(x['aux'])
        checked.append(dict(**x['record'],whole_prefix_environment_aux_exact=True,prefix_action_hash=digest(x['prefix_actions']),prefix_envkeys_hash=digest(x['prefix_envkeys'])))
    atomic_json(OUT/'BANK_FROZEN.json',dict(status='PASS',p1_file=str(exactpath),p1_sha256=sha(exactpath),p1_states=len(exact_records),records=checked,raw_states=len(records),excluded_rows=excluded,exclusion_reason='batch-shape dependent float auxiliary frontier mismatch; physical env exact, no outcome selection',raw_bank_sha256=sha(path),
        full_source_replay_exact=True,real_physical_P1=True,identity_relabeling=False,precarried_initialization=False,teleportation=False,
        limitation='map806; P0 historical allfar student Bank vs P1 native continued saved-state rule/route acquisition; source/exposure distributions differ, not generalization evidence',
        p0_bank='runs/carrier_mechanism_20261023/BANK_FROZEN.json',wall_seconds=time.time()-began,unix=time.time()))

if __name__=='__main__':main()
