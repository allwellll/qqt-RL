"""R7: reduce only return's share of accepted ordinary DAgger replay slots."""
import inspect
from scripts import phase_balanced_joint_replay as baseline


def build_choose():
    source = inspect.getsource(baseline.choose)
    old = 'ordinary_slots*3//4'
    assert source.count(old) == 1, 'unexpected phase quota implementation'
    scope = {}
    exec(compile(source.replace(old, 'ordinary_slots//2'),
                 '<retention-stage-allocation>', 'exec'), baseline.__dict__, scope)
    return scope['choose']


choose = build_choose()


def install(return_share):
    assert return_share == .5, 'only the preregistered single-factor arm is supported'
    from scripts import return_error_replay as replay
    replay.baseline_choose = choose
    return dict(passed=True, call_site='scripts.return_error_replay.baseline_choose',
                return_share=.5, baseline_return_share=.75,
                eligible='existing valid ordinary accepted route labels, unchanged',
                phase_quota_when_all_present=dict(returning=16, drop=8, stock=8,
                                                 navigation=30, opening=2),
                counts_per_current_or_history_half=64,
                absent_stage_fallback_unchanged=True,
                fifo_and_custody_proof_unchanged=True,
                reward_changed=False, loss_changed=False,
                query_budget_changed=False, in_memory_only=True)
