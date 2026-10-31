"""One surrogate coefficient; preserve all unqualified Actor rows exactly."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def fixture():
    c=jnp.zeros((3,116)).at[:,0].set(8).at[:,54].set(3)
    c=c.at[:,55].set(1).at[:,58].set(1).at[:,95].set(1)
    g=jnp.zeros((3,18)).at[:,9].set(1).at[:,13].set(1).at[:,14].set(1)
    return c,g,jnp.ones((3,5),bool),jnp.ones((3,3),bool)


def test_identity_and_no_coverage_have_exact_baseline_values_and_gradients():
    from scripts.retention_return_ppo import surrogate
    d=fixture();lp=jnp.array([-.4,.3,.7]);old=jnp.array([0.,0.,.2]);a=jnp.array([1.,-2.,.5])
    baseline=lambda x:jnp.maximum(-a*jnp.exp(x-old),-a*jnp.clip(jnp.exp(x-old),.9,1.1)).mean()
    for factor,data in [(1.,d),(.5,(d[0].at[:,58].set(0),*d[1:]))]:
        actual=lambda x:surrogate(x,old,a,*data,factor)
        np.testing.assert_array_equal(actual(lp),baseline(lp))
        np.testing.assert_array_equal(jax.grad(actual)(lp),jax.grad(baseline)(lp))


def test_qualified_rows_keep_original_denominator_and_clip_derivatives():
    from scripts.retention_return_ppo import surrogate
    d=list(fixture());d[0]=d[0].at[1,58].set(0)
    old=jnp.zeros(3);a=jnp.array([1.,-2.,.5]);scale=jnp.array([.5,1.,.5])
    for ratio in [[.8,1.,1.2],[1.05,.8,.95],[1.2,1.2,.8]]:
        lp=jnp.log(jnp.array(ratio))
        expected=lambda x:(jnp.maximum(-a*jnp.exp(x-old),-a*jnp.clip(jnp.exp(x-old),.9,1.1))*scale).mean()
        actual=lambda x:surrogate(x,old,a,*d,.5)
        np.testing.assert_array_equal(jax.jit(actual)(lp),expected(lp))
        np.testing.assert_allclose(jax.grad(actual)(lp),jax.grad(expected)(lp),atol=1e-7)
    np.testing.assert_array_equal(actual(lp),surrogate(lp,old,a,*pickle.loads(pickle.dumps(d)),.5))


@pytest.mark.parametrize('field,index,value',[
    ('c',0,0),('c',0,2),('c',0,3),('c',0,4),('c',0,6),
    ('c',1,1),('c',47,1),('c',54,0),('c',54,1),('c',54,2),('c',54,4),
    ('c',55,0),('c',58,0),('c',110,1),('c',95,0),
    ('g',9,0),('g',9,.5),('g',13,0),('g',14,0)])
def test_no_native_custody_or_current_certificate_cannot_attenuate(field,index,value):
    from scripts.retention_return_ppo import row_scale
    d=list(fixture());i=0 if field=='c' else 1;d[i]=d[i].at[:,index].set(value)
    np.testing.assert_array_equal(row_scale(*d,.5),jnp.ones(3))


def test_illegal_joint_legacy_layout_and_mixed_ability_set():
    from scripts.retention_return_ppo import row_scale
    c,g,m,b=fixture()
    np.testing.assert_array_equal(row_scale(c[:,:95],g,m,b,.5),jnp.ones(3))
    np.testing.assert_array_equal(row_scale(c,g,m.at[:,0].set(False),b,.5),jnp.ones(3))
    np.testing.assert_array_equal(row_scale(c,g,m,b.at[:,0].set(False),.5),jnp.ones(3))
    # A certified alternative ability remains legal as a joint set.
    c=c.at[:,97].set(1)
    np.testing.assert_array_equal(row_scale(c,g,m,b.at[:,0].set(False),.5),jnp.full(3,.5))


def test_patch_changes_only_surrogate_and_is_idempotent():
    from pathlib import Path
    import ast
    from scripts.retention_return_ppo import patched
    source=Path('scripts/je09_stage_h5c_ops.py').read_text();result=patched(source,.5)
    assert patched(result,.5)==result and source!=result
    old="pol=jnp.maximum(-ad[ii[:half]]*ratio,-ad[ii[:half]]*jnp.clip(ratio,.9,1.1)).mean()"
    assert old not in result
    ast.parse(result)
    assert result.count('surrogate(logp,old[ii[:half]],ad[ii[:half]],')==1
    assert result.split('    def loss')[0]==source.split('    def loss')[0]
    assert result.split('        value=')[1]==source.split('        value=')[1]


def test_runtime_install_preserves_globals_and_disk_source(monkeypatch):
    from pathlib import Path
    from scripts import je09_stage_h5c_ops as ops
    from scripts import je09_stage_h6_ops,je09_stage_h8_ops,je10_ops
    from scripts.retention_return_ppo import install
    original=ops.ppo;disk=Path('scripts/je09_stage_h5c_ops.py').read_bytes()
    for module in [ops,je09_stage_h6_ops,je09_stage_h8_ops,je10_ops]:
        monkeypatch.setattr(module,'ppo',original)
    candidate=install(.5)
    assert candidate is not original and candidate.__globals__ is original.__globals__
    assert install(.5) is candidate
    with pytest.raises(AssertionError):install(1.)
    assert Path('scripts/je09_stage_h5c_ops.py').read_bytes()==disk


def test_install_reaches_the_worker_ppo_call_site(monkeypatch):
    from scripts import je09_stage_h5c_ops as base
    from scripts import je09_stage_h6_ops as h6
    from scripts import je09_stage_h8_ops as h8
    from scripts import je10_ops as worker_ops
    from scripts.retention_return_ppo import install
    for module in [base,h6,h8,worker_ops]:
        monkeypatch.setattr(module,'ppo',base.ppo)
    candidate=install(.5)
    assert worker_ops.ppo is candidate
    assert h6.ppo is candidate and h8.ppo is candidate


def test_real_ppo_updates_identity_and_treatment_with_matching_critic(monkeypatch):
    import inspect
    import textwrap
    import optax
    from scripts import je09_stage_h5c_ops as ops
    from scripts.retention_return_ppo import patched
    def forward(p,architecture,o,s):
        count=o.shape[0]
        return (jnp.broadcast_to(p['actor']['m'],(count,5)),
                jnp.broadcast_to(p['actor']['b'],(count,3)),None,
                jnp.broadcast_to(p['critic'],(count,1)))
    monkeypatch.setattr(ops.train,'net_forward',forward)
    monkeypatch.setattr(ops.train,'hl_gauss_value_loss',lambda v,r:((v[:,0]-r)**2).mean())
    monkeypatch.setattr(ops,'factor_kl',lambda *a:jnp.zeros((a[0].shape[0],3)))
    monkeypatch.setattr(ops.chase_expert,'supervised_loss',lambda *a:(jnp.float32(0),jnp.zeros(2)))
    monkeypatch.setattr(ops.return_expert,'supervised_loss',lambda *a:(jnp.float32(0),jnp.zeros(2)))
    monkeypatch.setattr(ops.combat_expert,'supervised_loss',lambda *a:jnp.float32(0))
    c,g,m,b=fixture()
    # One rollout with four rows; real Actor/Critic permutations and updates.
    c,g,m,b=[jnp.concatenate([x,x[:1]],0)[None] for x in [c,g,m,b]]
    p=dict(actor=dict(m=jnp.arange(5,dtype=jnp.float32)/10,b=jnp.zeros(3)),critic=jnp.float32(.1))
    acts=jnp.array([[[0,0],[1,1],[2,2],[3,0]]])
    lp=jax.nn.log_softmax(p['actor']['m'])[acts[...,0]]+jax.nn.log_softmax(p['actor']['b'])[acts[...,1]]
    batch=(jnp.zeros((1,4,1)),g,acts,lp,jnp.array([[1.,-1.,2.,-2.]]),jnp.array([[1.,2.,3.,4.]]),(m,b),c)
    opt=optax.sgd(.01)
    def run(fn):
        return fn(p,opt.init(p),batch,jax.random.PRNGKey(8),p,opt,
                  dict(g0=p,stage1_teacher=p),2,axis_name=None)
    baseline=run(ops.ppo)
    outputs=[]
    for factor in [1.,.5]:
        scope={}
        exec(compile(patched(textwrap.dedent(inspect.getsource(ops.ppo)),factor),
                     '<ppo-test>','exec'),ops.ppo.__globals__,scope)
        outputs.append(run(scope['ppo']))
    for expected,actual in zip(jax.tree.leaves(baseline),jax.tree.leaves(outputs[0])):
        np.testing.assert_array_equal(actual,expected)
    assert not np.array_equal(outputs[1][0]['actor']['m'],baseline[0]['actor']['m'])
    np.testing.assert_array_equal(outputs[1][0]['critic'],baseline[0]['critic'])


def test_complete_rollout_evidence_uses_exact_gate_and_replica_normalization():
    from scripts.retention_return_ppo import replica_evidence
    c,g,m,b=fixture();c=c.at[2,58].set(0).at[1,81].set(-.1)
    actions=jnp.array([[0,0],[1,0],[2,0]])
    adv=jnp.array([1.,3.,-1.])
    out=jax.jit(replica_evidence)(c[None],g[None],m[None],b[None],adv[None],actions[None])
    np.testing.assert_array_equal(out,jnp.array([3,2,1,1,1,1]))


def test_worker_telemetry_is_passive_and_installs_without_disk_edits(monkeypatch):
    import ast
    import inspect
    import textwrap
    from pathlib import Path
    from scripts import je10_worker as worker
    from scripts.retention_return_ppo import patched_worker,install_worker_telemetry
    original=worker.worker;source=textwrap.dedent(inspect.getsource(original))
    modified=patched_worker(source)
    # Remove only the declared instrumentation; all existing statements,
    # including collection, RNG, PPO, optimizer and save, must be identical.
    old=ast.parse(source)
    # Telemetry also sits inside the existing loop; strip just its statements.
    class Strip(ast.NodeTransformer):
        def visit_Assign(self,node):
            target=ast.unparse(node.targets[0])
            if target in ['ppo_evidence',"row['certified_return_ppo_evidence']"]:return None
            return node
        def visit_Expr(self,node):
            if ast.unparse(node).startswith("row['certified_return_ppo_evidence'].update("):return None
            return node
    # Preserve the loop while removing telemetry from it.
    new=ast.parse(modified)
    new=Strip().visit(new)
    new.body[0].body=[s for s in new.body[0].body if not
        (isinstance(s,ast.ImportFrom) and s.module=='scripts.retention_return_ppo')
        and not (isinstance(s,ast.Assign) and ast.unparse(s.targets[0])=='return_ppo_device_evidence')]
    assert ast.dump(new)==ast.dump(old)
    disk=Path('scripts/je10_worker.py').read_bytes()
    monkeypatch.setattr(worker,'worker',original)
    install_worker_telemetry(worker)
    assert worker.worker is not original and worker.worker.__globals__ is original.__globals__
    assert Path('scripts/je10_worker.py').read_bytes()==disk
