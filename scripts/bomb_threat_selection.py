"""Versioned BEST and PBT gates. Neither gate stops healthy PPO updates."""
import math

GATE_VERSION='gate-amend-20261029-0240'


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _common(natural,parent,baselines_complete):
    failures=[]
    if not baselines_complete:failures.append('THREE_BASELINES_INCOMPLETE')
    g=natural.get('greedy',{});p=(parent or {}).get('greedy',{})
    if g.get('games')!=64:failures.append('ARGMAX_DENOMINATOR_NOT_64')
    if not parent:failures.append('MATCHED_PARENT_REFERENCE_MISSING')
    elif p.get('games')!=64:failures.append('PARENT_ARGMAX_DENOMINATOR_NOT_64')
    if parent:
        for field,offset,label in [('cage_kill_fraction',.1,'CAGE_KILL'),('self_deaths',0,'SELF_DEATH')]:
            a,b=g.get(field),p.get(field)
            if not _number(a) or not _number(b):failures.append(label+'_DENOMINATOR_MISSING')
            elif a>b+offset+1e-9:failures.append(label+'_REGRESSION')
    return failures,g,p


def hard_gate_failures(natural,parent,reward_monitor,baselines_complete):
    """Absolute BEST/publication gate. Revision explicitly requires lower draws."""
    failures,g,p=_common(natural,parent,baselines_complete)
    if not _number(g.get('zero_bomb_games')) or g['zero_bomb_games']>3:failures.append('ARGMAX_ZERO_BOMBS_GT_3')
    if not _number(g.get('bombs_per_game')) or g['bombs_per_game']<15:failures.append('ARGMAX_BOMBS_LT_15')
    if parent:
        a,b=g.get('draw_fraction'),p.get('draw_fraction')
        if not _number(a) or not _number(b):failures.append('DRAW_DENOMINATOR_MISSING')
        elif a>=b-1e-9:failures.append('DRAW_NOT_LOWER_THAN_PARENT')
    if reward_monitor.get('reward_share_gate_pass') is not True:failures.append('NATIVE_REWARD_SHARE_MISSING_OR_FAILED')
    return failures


def relative_gate_failures(natural,parent,reward_monitor,baselines_complete):
    """PBT selection only: no absolute zero-bomb or terminal-reward ceiling."""
    failures,g,p=_common(natural,parent,baselines_complete)
    if parent:
        for field,scale,label,lower in [('zero_bomb_games',1,'ARGMAX_ZERO_BOMBS_GT_PARENT',False),
                ('bombs_per_game',.8,'ARGMAX_BOMBS_LT_PARENT_80_PERCENT',True),
                ('draw_fraction',1,'DRAW_REGRESSION',False)]:
            a,b=g.get(field),p.get(field)
            if not _number(a) or not _number(b):failures.append(label+'_DENOMINATOR_MISSING')
            elif (a+1e-9<b*scale if lower else a>b*scale+1e-9):failures.append(label)
    share=reward_monitor.get('threat_reward_fraction_total')
    if not _number(share) or not 0<=share<=.3+1e-9:
        failures.append('RELATIVE_NATIVE_REWARD_SHARE_MISSING_OR_FAILED')
    return failures


def shaping_trend(points,reference):
    """Two consecutive same-lineage deltas, starting at the matched reference.

    Caller supplies chronological, distinct checkpoints from one split/seed.
    This reports a shaping-pause request; it never changes coefficients itself.
    """
    consecutive=0;previous=reference;comparisons=[]
    for point in points:
        fields=('bombs_per_game','causal_kills','win')
        known=previous is not None and all(_number(x.get(k)) for x in (point,previous) for k in fields)
        flag=bool(known and point['bombs_per_game']>previous['bombs_per_game'] and
                  point['causal_kills']<=previous['causal_kills'] and point['win']<=previous['win'])
        consecutive=consecutive+1 if flag else 0
        comparisons.append(dict(bombs_up_without_kills_or_wins=flag,known=known,consecutive=consecutive))
        previous=point
    return dict(pause_required=consecutive>=2,comparisons=comparisons,
        interpretation='two consecutive deltas versus previous evaluated checkpoint in same slot, split and seed; matched parent is initial reference')
