"""Passive argmax gates, decision amendment 2; never publishes or stops PPO."""
import math
GATE_VERSION='gate-amend2-zero-bomb-20261029-1420'

def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def result(failures,missing):
    return dict(status='INCOMPLETE' if missing else 'FAIL' if failures else 'PASS',
                failures=list(dict.fromkeys(failures)),missing=list(dict.fromkeys(missing)))

def evaluate(natural,parent,reward,split,references=None):
    g=natural.get('greedy',{});p=(parent or {}).get('greedy',{})
    failures=[];missing=[]
    def compare(field,label,test):
        a,b=g.get(field),p.get(field)
        if not number(a) or not number(b):missing.append(label+'_MISSING')
        elif not test(a,b):failures.append(label)
    if g.get('games')!=64:missing.append('ARGMAX_64_COMPLETE_GAMES_REQUIRED')
    if p.get('games')!=64:missing.append('MATCHED_PARENT_64_COMPLETE_GAMES_REQUIRED')
    compare('self_deaths','SELF_DEATH_GT_PARENT_PLUS_2',lambda a,b:a<=b+2)
    compare('cage_kill_fraction','CAGE_KILL_GT_PARENT_PLUS_10PP',lambda a,b:a<=b+.1+1e-9)
    share=reward.get('threat_reward_fraction_total')
    if not number(share):missing.append('NATIVE_REWARD_SHARE_MISSING')
    elif not 0<=share<=.3+1e-9:failures.append('SHAPING_SHARE_GT_30_PERCENT_OR_NEGATIVE')
    common_f=list(failures);common_m=list(missing)
    compare('zero_bomb_games','ZERO_BOMBS_GT_PARENT_PLUS_3',lambda a,b:a<=b+3)
    compare('bombs_per_game','BOMBS_LT_PARENT_80_PERCENT',lambda a,b:a+1e-9>=.8*b)
    compare('draw_fraction','DRAW_GT_PARENT_PLUS_5PP',lambda a,b:a<=b+.05+1e-9)
    relative=result(failures,missing)
    # The absolute comparison is meaningful only on matched holdout data.
    if split!='holdout':
        absolute=result([],['HOLDOUT_ARGMAX_AND_CATEGORICAL_64_REQUIRED'])
    else:
        failures=common_f;missing=common_m
        if natural.get('categorical',{}).get('games')!=64:missing.append('HOLDOUT_CATEGORICAL_64_COMPLETE_GAMES_REQUIRED')
        compare('win_minus_loss','WIN_MINUS_LOSS_LT_PARENT_PLUS_8',lambda a,b:a>=b+8)
        if not number(g.get('zero_bomb_games')):missing.append('ZERO_BOMBS_MISSING')
        elif g['zero_bomb_games']>7:failures.append('ZERO_BOMBS_GT_7')
        if not number(g.get('bombs_per_game')):missing.append('BOMBS_PER_GAME_MISSING')
        elif g['bombs_per_game']<20:failures.append('BOMBS_LT_20')
        compare('draw_fraction','DRAW_GT_PARENT',lambda a,b:a<=b+1e-9)
        compare('causal_kills','KILLS_LT_PARENT',lambda a,b:a>=b)
        compare('deaths','DEATHS_GT_PARENT_PLUS_5',lambda a,b:a<=b+5)
        for name in ('reference_transport1536','reference_selfplay3072'):
            ref=(references or {}).get(name,{}).get('greedy',{})
            a,b=g.get('win_minus_loss'),ref.get('win_minus_loss')
            if ref.get('games')!=64 or not number(a) or not number(b):missing.append(name.upper()+'_MATCHED_HOLDOUT_MISSING')
            elif a<=b:failures.append('WIN_MINUS_LOSS_NOT_ABOVE_'+name.upper())
        absolute=result(failures,missing)
    return dict(relative=relative,absolute=absolute,
                best=dict(status='INCOMPLETE' if absolute['status']=='PASS' else absolute['status'],
                          failures=absolute['failures'],missing=absolute['missing']+(['FRESH_SEED_256_PER_TIER_RECHECK_REQUIRED'] if absolute['status']=='PASS' else [])))
