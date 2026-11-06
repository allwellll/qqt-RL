"""Unit and anti-hack tests for cc_fountain_reward_v1 (CPU). usage: python -m scripts.cc_fountain_reward_v1_test
Writes runs/cc_train_20261105/training/league/FTN_REWARD_V1_TESTS.json."""
import inspect, os, subprocess, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax.numpy as j
from scripts import r3_items_reward as R
from scripts.population_run_state import atomic_json

before_transition = inspect.getsource(R.transition)
old_chain = R.chain_reward
from scripts import cc_fountain_reward_v1 as F
F.install()
after_transition = F.SOURCES['transition']
OUT = {}
def check(name, cond, detail):
    OUT[name] = dict(passed=bool(cond), detail=str(detail)); assert cond, (name, detail)
close = lambda a, b: abs(float(a) - float(b)) < 1e-6

# ---- CHAIN
k = j.arange(9)
check('chain_coef_k6', bool(j.allclose(R.chain_reward(k), .04 * j.minimum(k, 6))), R.chain_reward(k))
check('chain_near_x1_5', bool(j.allclose(R.chain_reward(k, near=True), .06 * j.minimum(k, 6))), R.chain_reward(k, near=True))
check('chain_max_event_036', close(R.chain_reward(j.array(10), near=True), .36), R.chain_reward(j.array(10), near=True))
check('chain_zero_without_cross_batch', float(R.chain_reward(j.array(0), near=True)) == 0, 'k=0')
for flag in ('bad', 'empty', 'home_stack', 'same_batch'):
    check(f'chain_void_{flag}', float(R.chain_reward(j.array(5), near=True, **{flag: True})) == 0, flag)
check('cross_batch_counter_covers_k6', 'for i in range(10)' in inspect.getsource(R.cross_batch_count), 'distinct batches counted up to 10 >= 6')

# ---- fountain (PRESSURE) pure step
def run(events, bad_at=()):
    """events: list of (t, batch, effective). returns pays, streaks."""
    c, lt, lb = j.int32(0), j.int32(-1000), j.int32(-1); pays = []; streaks = []
    for i, (t, b, e) in enumerate(events):
        p, c, lt, lb, s, _ = F.fountain_step(c, lt, lb, j.int32(t), j.int32(b), j.bool_(e), j.bool_(i in bad_at))
        pays.append(round(float(p), 6)); streaks.append(int(s))
    return pays, streaks, c
pays, streaks, _ = run([(10 * i, i, True) for i in range(7)])
check('fountain_streak_increments', pays == [.01, .02, .03, .04, .05, .05, .05] and streaks == [1, 2, 3, 4, 5, 6, 7], (pays, streaks))
pays, streaks, _ = run([(0, 0, True), (24, 1, True), (49, 2, True), (60, 3, True)])
check('fountain_gap_24_keeps_25_breaks', pays == [.01, .02, .01, .02], (pays, streaks))
pays, _, _ = run([(0, 0, True), (5, 0, True), (10, 0, True)])
check('fountain_same_batch_once', pays == [.01, 0, 0], pays)
pays, _, _ = run([(0, 0, True), (10, 1, True), (20, 2, True), (30, 3, True)], bad_at=(2,))
check('fountain_bad_resets_streak', pays[3] == .01, pays)
pays, _, c = run([(5 * i, i, True) for i in range(30)])
check('fountain_max_20_batches', sum(p > 0 for p in pays) == 20 and int(c) % F.PACK == 20, (pays, int(c)))
check('fountain_episode_sum', close(sum(pays), sum(.01 * min(n, 5) for n in range(1, 21))), sum(pays))
check('fountain_pack_no_overflow', F.FOUNTAIN_MAX < F.PACK and 31 * F.PACK + F.PACK < 2 ** 31, F.PACK)
check('fountain_restore_compatible_old_count', int(F.fountain_step(j.int32(7), j.int32(-1000), j.int32(-1), j.int32(0), j.int32(3), j.bool_(True), j.bool_(False))[1]) == 8 + F.PACK,
      'old pressure_count<=10 decodes as total, streak 0')

# ---- anti-hack (all via the factual gates in transition)
# A wall/empty corner self-chain: no novel near-enemy cover -> effective False -> fountain 0; useful False -> empty -> CHAIN 0.
pays, _, _ = run([(5 * i, i, False) for i in range(40)])
check('antihack_corner_spam_fountain_0', sum(pays) == 0, 'effective=False (no novel cell within 3 of enemy)')
check('antihack_corner_chain_0', float(R.chain_reward(j.array(6), empty=True)) == 0, 'useful=False -> empty')
check('antihack_far_enemy_gate', "effective=explosion&(novel&near).any((-2,-1))" in after_transition
      and "near=(j.abs(rr[None]-enemycell[:,0,None,None])+j.abs(cc[None]-enemycell[:,1,None,None])<=3)&after.alive[rows,enemy,None,None]" in after_transition,
      'fountain needs novel cover within 3 of live enemy')
check('antihack_home_stack_source_gate', "source_valid=resolved_own&((homefields[rows,seats]>3)|(enemy_home[:,None,None]<=3))" in after_transition,
      'bombs within 3 of own home are not sources unless enemy is within 3 of home')
check('antihack_post_death_0', "explosion=source_valid.any((-2,-1))&useful&active&before.alive[rows,enemy]" in after_transition, 'enemy dead before blast -> no attack credit')
check('antihack_bad_voids_queue', 'invalid=bad' in inspect.getsource(R.settle) and "queue=j.where(invalid[:,None,None],0.,queue)" in inspect.getsource(R.settle),
      'own death/trap voids all unpaid queued credit (30-tick quarantine)')
check('no_count_tick_idle_terms', not any(w in after_transition for w in ('idle', 'stationary', 'bomb_placed', 'tick_stock')), 'source scan')
removed = [l for l in before_transition.splitlines() if l not in after_transition.splitlines()]
added = [l for l in after_transition.splitlines() if l not in before_transition.splitlines()]
check('transition_diff_scoped', len(removed) == 7 and len(added) == 9 and all(any(w in l for w in ('CHAIN', 'PRESSURE', 'THREAT', 'explosion', 'pressure', 'fountain', 'chain_near', 'effective', 'fcount')) for l in added),
      dict(removed=removed, added=added))

# ---- payment caps / budget
s = R.state(1); big = j.full((1, 8), 1.)
s1, out = R.capped_payment(s, big)
check('attack_budget_090', close(out[0, 5:].sum(), F.ATTACK_BUDGET), out[0])
check('item_budget_unchanged_060', close(out[0, :5].sum(), .60), out[0])
check('slot_caps', bool((out[0] <= j.array(F.CAPS) + 1e-6).all()), out[0])
check('monotone_no_refill', float(R.capped_payment(s1, big)[1].sum()) == 0, 'second call')
check('inherited_attack_counts_in_budget', close(R.capped_payment(R.state(1), big, attack_gross=j.array([.50]))[1][0, 5:].sum(), .40), 'inherited .50')
neg = j.zeros((1, 8)).at[0, 6].set(-.05)
check('penalty_passes', close(R.capped_payment(R.state(1), neg)[1][0, 6], -.05), 'negative not capped')
# Release-bun / kill carrier / recapture loop: custody potential telescopes, attack credit bounded by the episode budget.
from scripts.bun_native_v2_custody_reward import graded_phi
phi = lambda h, c, l: float(graded_phi(j.array(h), j.array(c), j.array(l), j.array(.5), j.array(.5)))
check('loop_custody_telescopes_0', abs((phi(False, True, False) - phi(True, False, False)) + (phi(True, False, False) - phi(False, True, False))) < 1e-9, 0)
s = R.state(1); total = 0.; last = None
for loop in range(30):
    s, o = R.capped_payment(s, j.zeros((1, 8)).at[0, 5:].set(.1)); last = float(o[0, 5:].sum()); total += last
check('loop_attack_credit_bounded', total <= F.ATTACK_BUDGET + 1e-6 and last == 0, dict(total=total, last=last))
UPPER = dict(per_episode_attack_max=F.ATTACK_BUDGET, base_r3=.45, atk_v2=.60, extra_vs_base=round(F.ATTACK_BUDGET - .45, 2), extra_vs_atk_v2=round(F.ATTACK_BUDGET - .60, 2),
             slot_caps=dict(chain=.60, pressure=.40, threat=.45), fountain_uncapped_episode_sum=round(sum(.01 * min(n, 5) for n in range(1, 21)), 2), chain_event_max=.36)
check('upper_bound_documented', UPPER['extra_vs_base'] == .45, UPPER)

# ---- install_on_import does not import jax or the reward module
code = ("import sys; from scripts import cc_fountain_reward_v1 as F; F.install_on_import();"
        "assert 'jax' not in sys.modules and 'scripts.r3_items_reward' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)[:5];"
        "import os; os.environ['JAX_PLATFORMS']='cpu'; from scripts import r3_items_reward as R; assert R._cc_fountain_v1; print('ok')")
r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
check('install_on_import_lazy', r.returncode == 0 and r.stdout.strip() == 'ok', (r.returncode, r.stdout[-200:], r.stderr[-300:]))

atomic_json('runs/cc_train_20261105/training/league/FTN_REWARD_V1_TESTS.json',
            dict(status='PASS', version=F.VERSION, caps=F.CAPS, attack_budget=F.ATTACK_BUDGET, upper_bounds=UPPER, tests=OUT, unix=time.time()))
print('PASS', len(OUT), 'tests')
