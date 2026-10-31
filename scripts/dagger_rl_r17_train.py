"""R16 sequential optimizer/collector with only valid-goal qualification."""
from pathlib import Path
import copy
from scripts import combat_item_engine as M, je10_route_oracle as GO
from scripts.dagger_rl_route_goal_eligibility import guarded_teacher
from scripts.dagger_rl_goal_guard_collector import factory
from scripts.population_run_state import atomic_json, sha

ORIGINAL_CONFIGURE = M.configure
ORIGINAL_TEACHER = M.route_teacher
COUNTERS = {}


def install_goal_guard():
    def configure():
        ORIGINAL_CONFIGURE()
        M.route_teacher = guarded_teacher(ORIGINAL_TEACHER, M.E.inputs, GO.labels)
        M.E.safe_teacher = M.route_teacher
    M.configure = configure
    L.C.collector = lambda steps: factory(L.S, steps, legacy_teacher=ORIGINAL_TEACHER,
                                        stream='PPO', counters=COUNTERS)

    def auxiliary(steps, greedy=False):
        if greedy:
            raise ValueError('categorical student stream is frozen')
        # R16 auxiliary_choose(greedy=False) delegates exactly to E.choose.
        return factory(L.S, steps, legacy_teacher=ORIGINAL_TEACHER,
                       stream='auxiliary', counters=COUNTERS)
    L.auxiliary_collector = auxiliary
    L.commit = commit_with_goal_counts


def commit_with_goal_counts(*args, **kwargs):
    metrics = args[10] if len(args) > 10 else kwargs.get('metrics')
    if metrics is not None:
        metrics['goal_guard_telemetry'] = copy.deepcopy(COUNTERS)
    result = ORIGINAL_COMMIT(*args, **kwargs)
    update = args[1]
    if update == 0 and kwargs.get('formal', True):
        from scripts.terminal_checkpoint_restore import digest
        import pickle
        from scripts.dagger_rl_r17_run import PRIOR
        reference = PRIOR/'formal'/args[0].name/'u000/state.pkl'
        current = pickle.load((result/'state.pkl').open('rb'))
        control = pickle.load(reference.open('rb'))
        keys = ['params', 'adam', 'states', 'aux', 'keys', 'auxiliary']
        assert digest(tuple(current[k] for k in keys)) == digest(tuple(control[k] for k in keys))
        atomic_json(args[0]/'MATCHED_INITIAL_STATE.json', dict(passed=True,
            control_checkpoint=str(reference.relative_to(R)), control_state_sha256=sha(reference),
            state_sha256=sha(result/'state.pkl'), params_Adam_both_streams_exact=True))
    return result


source = Path(__file__).with_name('dagger_rl_r16_train.py').read_text()
for old, new in [
    ('scripts.dagger_rl_r16_run', 'scripts.dagger_rl_r17_run'),
    ("'scripts.dagger_rl_r16_train'", "'scripts.dagger_rl_r17_train'"),
    ('    BASE_BIND()\n', '    BASE_BIND()\n    install_goal_guard()\n'),
    ("return dict(joint_anchor=L.C.updater('anchor', optimizer),\n                dagger_anchor=",
     'return dict(dagger_anchor='),
    ("outputs['joint_anchor']", "outputs['dagger_anchor']"),
    ('total_executed_optimizer_steps=18', 'total_executed_optimizer_steps=12'),
    ('actual_branch_optimizer_steps=6, reference_optimizer_steps=6, fresh_restore_optimizer_steps=6',
     'actual_branch_optimizer_steps=4, reference_optimizer_steps=4, fresh_restore_optimizer_steps=4'),
]:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
source = source.replace('R16', 'R17').replace('三损失分支', '两损失分支')
# Prebind before retention_loop copies r12_train globals into its exec scope.
anchor = 'formal_function(L, bind=bind'
assert source.count(anchor) == 1
source = source.replace(anchor, 'bind()\n    formal_function(L, bind=bind')
# R16 dispatch is below its final main guard; set the commit reference first.
source = source.replace("if __name__ == '__main__':", "ORIGINAL_COMMIT = L.commit\n\nif __name__ == '__main__':")
exec(compile(source, '<R17-frozen-sequential-training>', 'exec'), globals())
