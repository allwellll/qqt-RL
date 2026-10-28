from tests.external_artifacts import requires_artifacts
import jax
import jax.numpy as j
import numpy as np
from scripts import combat_mix_legacy_batched as B

def test_direct_collector_preserves_original_batch_seats_rng_and_payload(monkeypatch):
    calls=[]
    def collect(params,st,aux,key,pool,frozen,course,pfsp,steps,**kwargs):
        calls.append((st,key,steps));n=len(st['t']);ticks=steps
        a=j.arange(ticks*n*2).reshape(ticks,n,2)
        data=j.ones((ticks,n));ctx=j.zeros((ticks,n,120)).at[...,92].set(3).at[...,93].set(7).at[...,94].set(1)
        return st,aux,key+1,(data,data,a,data,data,data,(data,data),ctx),(j.ones((10,4,65)),j.ones((11,6)),j.ones(83),j.array([.1,.2]))
    monkeypatch.setattr(B.ops,'collect',collect)
    st={'t':j.array([0,1,2,3])};aux={};key=j.arange(8,dtype=j.uint32).reshape(4,2)
    end,batch,diag=B.raw(2,{},st,aux,key,{}, {},j.ones(11),j.int32(161))
    assert len(calls)==1 and len(calls[0][0]['t'])==4
    assert np.array_equal(calls[0][1],key[0])
    assert np.array_equal(batch[2],j.arange(16).reshape(2,4,2))
    assert np.array_equal(end[2],j.broadcast_to(key[0]+1,key.shape))
    assert int(diag['executed_source_live_ticks'][3])==8
    assert int(diag['executed_league_member_live_ticks'][7])==8


@requires_artifacts('runs/combat_mix_v1/protocols/legacy_mix_s2026103051.json')
def test_real_original_full_batch_and_adapter_have_exact_next_step():
    import pickle,json
    from pathlib import Path
    from scripts import ppo_scale_longrun_train as P
    P.T.configure();P.C.install_provenance()
    doc=json.loads(Path('runs/combat_mix_v1/protocols/legacy_mix_s2026103051.json').read_text())
    x=pickle.load((Path(doc['sources']['2026103051']['checkpoint'])/'state.pkl').open('rb'))
    st,aux,keys=jax.tree.map(lambda a:j.asarray(a[0,:6]),(x['states'],x['aux'],x['keys']))
    aux=B.base_aux(aux)
    frozen=pickle.load(Path(doc['legacy_inputs']['frozen']['path']).open('rb'))
    frozen={k:frozen[k] for k in ('g0','stage1_teacher')};frozen['parent']=x['params']['actor']
    pool=B.initial_pool(pickle.load(Path(doc['legacy_inputs']['initial_pool']['path']).open('rb')),x['params']['actor'])
    pf=B.ops.pfsp_weights(j.zeros((11,6)))
    def compare(q,s,a,k):
        direct=B.ops.collect(q,s,a,k[0],pool,frozen,B.course(),pf,1,
            assist_probability=.1,expert_hazard_mode='local_safe',
            bridge_config=dict(window=120,scale=2.),return_config=dict(window=120),
            chase_stage=j.int32(0),chase_supervision=False,dagger_round=j.int32(161))
        adapted=B.raw(1,q,s,a,k,pool,frozen,pf,j.int32(161))
        return B.adapt_result(direct,6),adapted
    original,got=jax.jit(compare)(x['params'],st,aux,keys)
    left,lt=jax.tree.flatten(original);right,rt=jax.tree.flatten(got)
    assert lt==rt
    for a,b in zip(left,right):
        assert np.isfinite(np.asarray(a)).all()
        assert np.array_equal(np.asarray(a),np.asarray(b))
    assert got[1][2].shape==(1,6,2)
