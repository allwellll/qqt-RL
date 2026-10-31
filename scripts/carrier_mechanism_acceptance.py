"""Independent atomic-checkpoint and matched-initialization acceptance, no training."""
import json,pickle
from scripts.carrier_mechanism_run import B,R,read,authorize
from scripts.population_run_state import atomic_json,sha
from scripts.terminal_checkpoint_restore import digest


def verify():
    authorize();protocol=read(B/'PROTOCOL.json')['C'];assert read(B/'C_TRAINING_COMPLETE.json')['passed']
    execution=read(B/'C_EXECUTION_SOURCE.json');assert execution['protocol_sha256']==sha(B/'PROTOCOL.json')
    for name,value in execution['sources'].items():assert sha(R/name)==value
    parent=pickle.load((B/'inputs/parent.pkl').open('rb'));p0=digest(parent['params']);a0=digest(parent['adam'])
    control=pickle.load((B/'C/no_update/u000/state.pkl').open('rb'))
    assert digest(control['params'])==p0 and digest(control['adam'])==a0
    assert digest(pickle.load((B/'C/no_update/u000/policy.pt').open('rb')))==digest(control['params']['actor'])
    result=[];checkpoint_hashes={}
    per_update=protocol['rollout_envs']*protocol['rollout_steps'];optimizer_per_update=per_update//protocol['minibatch']*protocol['ppo_epochs']
    for seed in protocol['paired_seeds']:
        shared_initial_environment=None
        for arm in protocol['arms']:
            base=B/'C'/f'{arm}_s{seed}';receipt=read(base/'COMPLETE.json');initial=pickle.load((base/'u000/state.pkl').open('rb'))
            assert digest(initial['params'])==p0 and digest(initial['adam'])==a0
            env=digest((initial['states'],initial['aux'],initial['keys']))
            if shared_initial_environment is None:shared_initial_environment=env
            assert env==shared_initial_environment,'paired arms differ before first update'
            for update in protocol['save_updates']:
                cp=base/f'u{update:03d}';committed=read(cp/'COMMITTED.json')
                assert committed['seed']==seed and committed['arm']==arm and committed['update']==update
                assert committed['state_sha256']==sha(cp/'state.pkl') and committed['policy_sha256']==sha(cp/'policy.pt')
                assert committed['environment_steps']==update*per_update and committed['optimizer_steps']==update*optimizer_per_update
                checkpoint_hashes[str(cp.relative_to(B))]=dict(state=committed['state_sha256'],actor=committed['policy_sha256'])
            assert receipt['atomic_state_sha256']==read(base/'u016/COMMITTED.json')['state_sha256']
            rows=[json.loads(line) for line in (base/'updates.jsonl').read_text().splitlines()]
            assert [r['update'] for r in rows]==list(range(1,protocol['updates']+1))
            assert all(r['finite'] and r['update_l2']['actor']>0 for r in rows)
            assert all((r['update_l2']['critic']==0 if arm=='DAgger-only' else r['update_l2']['critic']>0) for r in rows)
            final=pickle.load((base/'u016/state.pkl').open('rb'))
            assert digest(pickle.load((base/'u016/policy.pt').open('rb')))==digest(final['params']['actor'])
            if arm=='DAgger-only':
                assert digest(final['params']['critic'])==digest(parent['params']['critic'])
                for moment in ['mu','nu']:
                    assert digest(getattr(final['adam'][1][0],moment)['critic'])==digest(getattr(parent['adam'][1][0],moment)['critic'])
                assert int(final['adam'][1][0].count)==int(parent['adam'][1][0].count)+protocol['updates']*optimizer_per_update
            result.append(dict(arm=arm,seed=seed,finite_nonzero=True,initial_environment_aux_RNG_digest=env,
                environment_steps=receipt['environment_steps'],optimizer_steps=receipt['optimizer_steps']))
    atomic_json(B/'C_ATOMIC_ACCEPTANCE.json',dict(passed=True,runs=result,checkpoint_hashes=checkpoint_hashes,
        common_parent_parameter_digest=p0,common_parent_Adam_digest=a0,paired_initial_env_aux_RNG_exact=True,
        dagger_critic_parameters_and_moments_frozen=True,shared_Adam_clock_advances=True,
        all_final_Actor_exports_match_live_states=True,
        training_environment_steps=sum(r['environment_steps'] for r in result),
        optimizer_steps=sum(r['optimizer_steps'] for r in result),old_continuous_resume=False))


if __name__=='__main__':verify()
