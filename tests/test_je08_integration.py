"""JE08 production integration and successor isolation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAXBOMB_RULE='bun')
import jax
import jax.numpy as jnp
import numpy as np
from scripts import je08_bun_ops as ops,je07_bun_ops as previous,je08_bun_recipe as recipe
from tests.native_bun_fixtures import fixture


def test_plan_predecessor_shape_assistance_and_namespace():
    p=recipe.plan()
    assert 'je08' in str(recipe.BASE) and 'je07' in str(recipe.PREDECESSOR)
    assert p['combat_fraction']==.45 and p['carry_assist']['probabilities']==[.7,.5,.3,.1]
    assert p['total_fresh_ticks']==8589934592 and p['updates_per_generation']==128
    assert p['evaluation']['assist_enabled'] is False
    assert recipe.SHAPE==[4,2048,256,4096]
    assert p['envs_per_device']*4==8192 and p['gradient_accumulation']==2
    assert p['envs_per_device']*p['steps']//(2048*p['gradient_accumulation'])==128
    assert p['handoff']['requires_training_complete'] is False
    assert p['potential']['drop_coefficient']==.05 and p['potential']['carry_coefficient']==.0625


def test_zero_reward_exact_je07_and_no_unattributed_own_drop():
    from jax_bomb import je07_bun_env as env
    s=fixture(dict(pos=[[5.5,5.5],[5.5,9.5]],loose=[[5,6,1,1]]))
    nxt,_,info=env.step(s,jnp.array([[3,0],[4,0]]),jax.random.PRNGKey(1),False,True)
    batch=lambda x:jax.tree.map(lambda y:y[None],x)
    a=ops.reward_state(1)
    r=ops.combined_reward(batch(s),batch(nxt),batch(info),a,jnp.ones(1,bool),coefficient=0)
    base=previous.combined_reward(batch(s),batch(nxt),batch(info),{k:v for k,v in a.items() if k not in ('potential','routing')},jnp.ones(1,bool))
    np.testing.assert_array_equal(r[0],base[0])
    _,_,_,m=ops.combined_reward(batch(s),batch(nxt),batch(info),a,jnp.ones(1,bool))
    assert not np.asarray(m['drop_approach_net']).any()


def test_handoff_safe_pause_without_completion_and_no_gpu_takeover(tmp_path,monkeypatch):
    monkeypatch.setattr(recipe,'PREDECESSOR',tmp_path)
    monkeypatch.setattr(recipe,'gpu_owners',lambda:{i:[] for i in range(4)})
    assert recipe.predecessor_finished()
    assert not (tmp_path/'TRAINING_COMPLETE.json').exists()
    monkeypatch.setattr(recipe,'gpu_owners',lambda:{0:[1],1:[],2:[],3:[]})
    assert not recipe.predecessor_finished()
    monkeypatch.setattr(recipe,'gpu_owners',lambda:{i:[] for i in range(4)})
    recipe.atomic_json(tmp_path/'TRAIN_LAUNCH.json',dict(pid=1))
    monkeypatch.setattr(recipe,'process_matches',lambda pid,script:True)
    assert not recipe.predecessor_finished()


def test_runtime_globals_and_sealed_dependencies():
    import builtins,importlib,symtable
    from pathlib import Path
    for kind in ['recipe','worker','evaluator','authority','supervisor','cpu_smoke']:
        module=importlib.import_module('scripts.je08_bun_'+kind)
        root=symtable.symtable(Path(module.__file__).read_text(),module.__file__,'exec')
        available=set(vars(module))|set(vars(builtins))|{s.get_name() for s in root.get_symbols() if s.is_assigned()}
        tables=[root]
        while tables:
            table=tables.pop();tables.extend(table.get_children())
            for symbol in table.get_symbols():
                if symbol.is_global() and symbol.is_referenced():assert symbol.get_name() in available,(kind,symbol.get_name())
    assert all(Path(s).exists() for s in recipe.SOURCES)


def test_real_attributed_drop_loose_motion_pickup_and_death_clawback():
    from jax_bomb import je07_bun_env as env
    s=fixture(dict(pos=[[5.5,4.5],[5.5,6.5]],carry=[-1,1],trap=[0,1]))
    s=s._replace(trapped_source=jnp.array([[False,False],[True,False]]),trapped_causal=jnp.array([[False,False],[True,False]]))
    batch=lambda q:jax.tree.map(lambda x:x[None],q)
    @jax.jit
    def tick(s,aux,action):
        nxt,_,info=env.step(s,jnp.stack([jnp.array([action,0]),jnp.array([4,0])]),jax.random.PRNGKey(2),False,True)
        r,a,parts,m=ops.combined_reward(batch(s),batch(nxt),batch(info),aux,jnp.ones(1,bool))
        return nxt,a,m,r,info
    s,a,m,_,info=tick(s,ops.reward_state(1),4)
    assert info['drop_credit'][1]==0 and a['routing']['target'][0,1,1]==6
    total=0
    for _ in range(20):
        s,a,m,r,info=tick(s,a,3)
        total+=float(m['drop_approach_net'][0,0])
        if s.bun_carried[0]==1:
            assert m['drop_approach_net'][0,0]==0 and m['carry_home_net'][0,0]==0
            assert m['drop_bonus'][0,0]==8
            break
    assert s.bun_carried[0]==1 and total>0 and total<=2.
    assert not a['potential']['drop']['active'][0,0]
    s=s._replace(trapped=s.trapped.at[0].set(1),trapped_source=jnp.zeros((2,2),bool),trapped_causal=jnp.zeros((2,2),bool))
    s,a,m,r,info=tick(s,a,4)
    assert info['death'][0] and m['potential_clawback'][0,0]<=-total+.0001 and m['drop_clawback'][0,0]==-8


def test_actual_topology_change_at_stationary_position_pays_no_delta():
    from jax_bomb import je07_bun_env as env
    s=fixture(dict(pos=[[5.5,5.5],[5.5,9.5]],carry=[1,-1]))
    batch=lambda q:jax.tree.map(lambda x:x[None],q)
    nxt,_,info=env.step(s,jnp.array([[4,0],[4,0]]),jax.random.PRNGKey(2),False,True)
    _,a,_,_=ops.combined_reward(batch(s),batch(nxt),batch(info),ops.reward_state(1),jnp.ones(1,bool))
    changed=nxt._replace(core=nxt.core._replace(brick=nxt.brick.at[5,6].set(True)))
    r,a,parts,m=ops.combined_reward(batch(nxt),batch(changed),batch(info),a,jnp.ones(1,bool))
    assert m['carry_home_net'][0,0]==0 and m['potential_topology_rebase'][0,0]>0


def test_accumulated_ppo_matches_combined_batch_one_adam_step(monkeypatch):
    """Both microbatches use the same parameters before a single Adam step."""
    import optax
    from jax_bomb import jax_train as train
    def forward(params,arch,obs,gv):
        actor=params.get('actor',params);critic=params.get('critic')
        value=obs@critic['value'] if critic is not None else jnp.zeros((obs.shape[0],1))
        return obs@actor['move'],obs@actor['ability'],jnp.zeros(obs.shape[0]),value
    monkeypatch.setattr(train,'net_forward',forward)
    monkeypatch.setattr(train,'hl_gauss_value_loss',lambda logits,target:jnp.mean((logits[:,0]-target)**2))
    key=jax.random.PRNGKey(808)
    params=dict(actor=dict(move=jnp.arange(15,dtype=jnp.float32).reshape(3,5)/100,ability=jnp.arange(9,dtype=jnp.float32).reshape(3,3)/100),critic=dict(value=jnp.ones((3,1))*.1))
    obs=jax.random.normal(key,(2,4,3));gv=jnp.zeros((2,4,24));actions=jnp.zeros((2,4,2),jnp.int32)
    batch=(obs,gv,actions,jnp.full((2,4),-2.),jnp.arange(8,dtype=jnp.float32).reshape(2,4),jnp.ones((2,4)),(jnp.ones((2,4,5),bool),jnp.ones((2,4,3),bool)),jnp.zeros((2,4,2)))
    opt=optax.adam(7.5e-5);adam=opt.init(params);frozen=dict(g0=params['actor'],stage1_teacher=params['actor'])
    run=lambda mb,acc:jax.jit(lambda:ops.ppo(params,adam,batch,key,params,opt,frozen,mb,axis_name=None,gradient_accumulation=acc))()
    accumulated=run(8,2);combined=run(16,1)
    assert int(accumulated[1][0].count)==1
    for a,b in zip(jax.tree.leaves(accumulated),jax.tree.leaves(combined)):
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
