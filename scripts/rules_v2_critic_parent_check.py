"""CPU-only preparation check on the real parent and genuine old-rule states.

This is not a new-rule probe or a GPU smoke. It never changes a checkpoint.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json
from pathlib import Path
import pickle
import time
import numpy as np
import jax
from scripts.population_run_state import atomic_json,sha


def main():
    from scripts import expert_reward_alignment_fullmatch as F
    from scripts import rules_v2_critic as C
    from scripts.terminal_checkpoint_restore import digest
    base=Path('runs/rules_v2_critic_20261027/v1')
    implementation=Path('runs/authorized_campaign_20261026/protocols/rules_v2_critic_ab_v1_implementation01.json')
    protocol=json.loads(implementation.read_text());source=Path(protocol['parent']['checkpoint'])
    assert jax.default_backend()=='cpu'
    assert sha(source/'state.pkl')==protocol['parent']['state_sha256']
    assert sha(source/'policy.pt')==protocol['parent']['policy_sha256']
    bank=Path('runs/authorized_campaign_20261026/natural_bank_coverage_v1/batches/parent_s2026102751_u064_categorical_00000.pkl')
    proof=json.loads(bank.with_suffix('.json').read_text());assert sha(bank)==proof['sha256']
    with bank.open('rb') as f:records=pickle.load(f)
    if not records:raise ValueError('No causal old-rule probe states')
    assert all(r['record']['next_step_parity'] for r in records)
    F.T.configure();F.provenance.install_provenance()
    states=jax.tree.map(lambda *x:np.concatenate(x,0),*(r['states'] for r in records))
    aux=jax.tree.map(lambda *x:np.concatenate(x,0),*(r['aux'] for r in records))
    obs,gv,mm,bm=F.E.inputs(states);ledger=C.ledger_features(aux)
    with (source/'state.pkl').open('rb') as f:saved=pickle.load(f)
    params=saved['params'];original_digest=digest(params['actor'])
    # Compare identical execution contexts. CPU bf16 eager dispatch and a
    # fused JIT can differ materially even for the unchanged parent itself.
    parent_eager=F.E.prediction(params,states)
    parent=jax.jit(lambda p,o,g:F.E.train.net_forward(p,'transformer',o,g))(params,obs,gv)
    jax.block_until_ready(parent)
    execution_differences=[float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
                          for a,b in zip(parent,parent_eager[:4])]
    results=[]
    for arm in C.ARMS:
        migrated=C.migrate(params,arm);adam=C.migrate_adam(saved['adam'],migrated)
        actual=jax.jit(C.forward)(migrated,obs,gv,ledger);jax.block_until_ready(actual)
        for index in (0,1):np.testing.assert_array_equal(actual[index],parent[index])
        eager_actual=C.forward(migrated,obs,gv,ledger)
        for index in (0,1):np.testing.assert_array_equal(eager_actual[index],parent_eager[index])
        assert digest(migrated['actor'])==original_digest
        logits_equal=True
        action_parent=np.stack((np.argmax(np.where(mm,parent[0],-1e9),-1),np.argmax(np.where(bm,parent[1],-1e9),-1)),-1)
        action_actual=np.stack((np.argmax(np.where(mm,actual[0],-1e9),-1),np.argmax(np.where(bm,actual[1],-1e9),-1)),-1)
        np.testing.assert_array_equal(action_parent,action_actual)
        # The new Critic leaves must have a valid Adam structure, without
        # changing source counts or any pre-existing moment/weight leaf.
        def adam_states(node):
            if hasattr(node,'mu') and hasattr(node,'nu'):return [node]
            if isinstance(node,(list,tuple)):return [v for child in node for v in adam_states(child)]
            return []
        for old,new in zip(adam_states(saved['adam']),adam_states(adam)):
            assert int(old.count)==int(new.count)
            for name in ('mu','nu'):
                before=getattr(old,name);after=getattr(new,name)
                assert digest(before['actor'])==digest(after['actor'])
                for key in before['critic']:
                    assert digest(before['critic'][key])==digest(after['critic'][key])
                for key in set(after['critic'])-set(before['critic']):
                    assert all(np.count_nonzero(x)==0 for x in jax.tree.leaves(after['critic'][key]))
        results.append(dict(arm=arm,actor_logits_exact=logits_equal,eager_actor_logits_exact=True,
            compiled_actor_logits_exact=True,greedy_joint_actions_exact=True,
            actor_weights_digest=original_digest,Actor_Adam_exact=True,
            all_existing_Critic_Adam_leaves_exact=True,new_Critic_Adam_leaves_zero=True,
            critic_initial_value_max_abs_diff=float(np.max(np.abs(np.asarray(actual[2])-np.asarray(parent[2])))),
            states=len(records),critic_ledger_dim=len(C.LEDGER_FIELDS) if arm!=C.ARMS[0] else 0))
    atomic_json(base/'REAL_PARENT_CPU_ACCEPTANCE.json',dict(passed=True,
        scope='old-rule genuine states and real parent only; not G2/new-rule validation or GPU smoke',
        source_state_sha256=sha(source/'state.pkl'),source_bank=str(bank),source_bank_sha256=sha(bank),
        implementation_sha256=sha(implementation),code_sha256=sha('scripts/rules_v2_critic.py'),
        unchanged_parent_CPU_eager_vs_jit_max_abs_differences=execution_differences,
        comparison='like-for-like eager and JIT separately; not CPU/GPU numerical parity',
        rows=results,rule_patch_applied=False,gpu_used=False,unix=time.time()))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
