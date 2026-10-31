import copy
import numpy as np
import jax
import jax.numpy as j
import optax
from scripts import bomb_threat_iter2_recipe as R
from scripts import bomb_threat_reward as old


def test_warmup_gate_and_exact_ramp_endpoints():
    assert R.actor_ramp(0)==0 and R.actor_ramp(1)==.025 and R.actor_ramp(40)==1
    np.testing.assert_allclose([R.entropy(0,True),R.entropy(320,True)],[.005,.0015],rtol=1e-6)
    assert not R.warmup_ready([{'value_loss':i,'explained_variance':0} for i in range(19)])['passed']
    assert R.warmup_ready([{'value_loss':1.,'explained_variance':.31}]*20)['passed']
    assert R.warmup_ready([{'value_loss':1.,'explained_variance':0}]*20)['plateau']
    assert not R.warmup_ready([{'value_loss':100-i*4.,'explained_variance':0} for i in range(20)])['passed']


def test_warmup_actor_exact_and_real_critic_update_then_both_update():
    def loss(p,b,ent):
        value=((p['critic']['w']-b)**2).mean()
        return ((p['actor']['w']-b)**2).mean()+value,j.array([0.,value,0.,0.,0.])
    opt=optax.adam(.01);p={'actor':{'w':j.array([1.,2.])},'critic':{'w':j.array([1.,2.])}}
    adam={k:opt.init(v) for k,v in p.items()};before=copy.deepcopy(p)
    update=R.minibatch_step(opt,loss)
    p,adam,_=update(p,adam,j.zeros(2),.005,0.,True)
    np.testing.assert_array_equal(p['actor']['w'],before['actor']['w'])
    assert np.linalg.norm(p['critic']['w']-before['critic']['w'])>0
    assert int(adam['actor'][0].count)==0 and int(adam['critic'][0].count)==1
    before=copy.deepcopy(p);p,adam,_=update(p,adam,j.zeros(2),.005,.025,False)
    assert all(np.linalg.norm(p[k]['w']-before[k]['w'])>0 for k in p)


def test_parameterized_settlement_default_bitwise_and_boost_caps():
    e=dict(enemy_cell=j.array([1]),old_cell=j.array([1]),active=j.array([True]),bad=j.array([False]),terminal=j.array([False]),
        placed=j.array([True]),safe=j.array([True]),threat=j.array([True]),chain=j.array([True]),old_id=j.array([1]),
        owned_removed=j.ones((1,13,15),bool),improved=j.array([True]),settled=j.array([True]))
    s=old.state(1);a=old.settle(e,s);b=R.settlement(1.,.2)(e,s)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    n,pay,m=R.settlement(5.,.6)(e,s);np.testing.assert_allclose(pay,[.35],atol=1e-7)
    e['enemy_cell']=j.array([2]);e['old_id']=j.array([2]);e['owned_removed']=j.zeros((1,13,15),bool)
    n,pay,m=R.settlement(5.,.6)(e,n);assert float(n['gross'][0])<=np.float32(.6)


def test_existing_owner_switch_is_authorized_slot_only_and_never_repeats(tmp_path,monkeypatch):
    import json
    from scripts import bomb_threat_iter2_schedule as S
    base=tmp_path/'runs/bomb_threat_v1';(base/'iter2').mkdir(parents=True)
    (base/'iter2/SWITCH_DECISION.json').write_text(json.dumps(dict(status='AUTHORIZED_READY',slots={'4':'A1','7':'A2','1':'A3'})))
    monkeypatch.setattr(S,'BASE',base)
    assert S.switch_request(0,{'run':'old'}) is None
    assert S.switch_request(4,{'run':'old'})['target_arm']=='A1'
    assert S.switch_request(4,{'run':'new','iter2_arm':'A1'}) is None
