"""Causal, reject-first fragment attribution for H6."""
from enum import IntEnum


class Fragment(IntEnum):
    AMBIGUOUS = 0
    COMBAT_CONTROL = 1
    A_LOOSE_RECOVERY = 2
    B_CAGE_STOCK = 3
    CARRY_RETURN = 4
    CAGE_DELIVERY = 5


def current_fragment(*, terminal, learner_alive, learner_trapped, teammate_harmed,
                     enemy_alive, enemy_trapped, attributed_window, carrying_enemy,
                     loose_attributed, stock_in_enemy_cage, at_own_cage,
                     reachable, legal_safe_actions, provenance_clear,
                     glue_only=False, banana_only=False):
    """Classify only current public state; ambiguity or unsafe proposals reject.

    This is intentionally not inferred from episode outcome. The caller must
    provide native event provenance and a verifier-certified action set.
    """
    if (terminal or not learner_alive or learner_trapped or teammate_harmed
            or not provenance_clear or not reachable or not legal_safe_actions):
        return Fragment.AMBIGUOUS
    if carrying_enemy:
        return Fragment.CAGE_DELIVERY if at_own_cage else Fragment.CARRY_RETURN
    window = (not enemy_alive) or enemy_trapped or attributed_window
    if not window:
        return Fragment.COMBAT_CONTROL
    if loose_attributed:
        return Fragment.A_LOOSE_RECOVERY
    if stock_in_enemy_cage:
        return Fragment.B_CAGE_STOCK
    # The window exists but the objective state has not appeared yet. Do not
    # infer a future drop or inventory pickup from the episode's later result.
    if glue_only or banana_only:
        return Fragment.AMBIGUOUS
    return Fragment.AMBIGUOUS


def verify_teacher(proposed, legal, rollout_checks):
    """Accept a set only when each proposed action has a complete safe witness."""
    proposed = set(proposed)
    legal = set(legal)
    required = {'no_self_death', 'no_self_trap', 'no_teammate_harm',
                'no_route_block', 'time_expanded_hazard_clear',
                'native_transition_consistent'}
    if not proposed or not proposed.issubset(legal):
        return False, 'empty_or_illegal_action_set'
    if any(not isinstance(rollout_checks.get(action), dict) for action in proposed):
        return False, 'missing_counterfactual_rollout'
    for action in proposed:
        checks = rollout_checks[action]
        missing = required - set(checks)
        if missing:
            return False, 'incomplete_counterfactual:' + ','.join(sorted(missing))
        failed = [name for name in required if checks[name] is not True]
        if failed:
            return False, 'counterfactual_rejected:' + ','.join(sorted(failed))
    return True, 'accepted'
