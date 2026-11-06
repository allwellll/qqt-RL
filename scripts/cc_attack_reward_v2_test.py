"""Unit and anti-hack tests for cc_attack_reward_v2 (CPU). usage: python -m scripts.cc_attack_reward_v2_test
Writes runs/cc_train_20261105/training/league/ATK_REWARD_V2_TESTS.json."""
import inspect, os, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax.numpy as j
from scripts import r3_items_reward as R
from scripts.population_run_state import atomic_json

before_transition = inspect.getsource(R.transition)
old_chain = R.chain_reward
from scripts.cc_attack_reward_v2 import install, THREAT, PRESSURE, CHAIN, CAPS, ATTACK_BUDGET, SOURCES
install()
after_transition = SOURCES['transition']
OUT = {}
def check(name, cond, detail):
    OUT[name] = dict(passed=bool(cond), detail=str(detail)); assert cond, (name, detail)

k = j.arange(7)
check('chain_coef', bool(j.allclose(R.chain_reward(k), CHAIN * j.minimum(k, 4))), R.chain_reward(k))
check('chain_old_ratio_2x', bool(j.allclose(R.chain_reward(k), 2 * old_chain(k))), 'v2 = 2x v1 per cross-batch bomb')
for flag in ('bad', 'empty', 'home_stack', 'same_batch'):
    check(f'chain_void_{flag}', float(R.chain_reward(j.array(3), **{flag: True})) == 0, flag)
diff = [l for l in after_transition.splitlines() if l not in before_transition.splitlines()]
check('transition_only_two_coefficients_changed', len(diff) == 2 and all(('PRESSURE' in l or 'THREAT' in l) for l in diff), diff)
check('no_count_tick_idle_terms', not any(w in after_transition for w in ('idle', 'stationary', 'bombs_placed*', 'tick_stock')), 'source scan')

s = R.state(1)
big = j.full((1, 8), 1.)
s1, out = R.capped_payment(s, big)
attack = float(out[0, 5:].sum()); item = float(out[0, :5].sum())
check('attack_budget_060', abs(attack - ATTACK_BUDGET) < 1e-6, attack)
check('item_budget_unchanged_060', abs(item - .60) < 1e-6, item)
check('slot_caps', bool((out[0] <= j.array(CAPS) + 1e-6).all()), out[0])
s2, out2 = R.capped_payment(s1, big)
check('monotone_no_refill', float(out2.sum()) == 0, out2)
s3, out3 = R.capped_payment(R.state(1), big, attack_gross=j.array([.30]))
check('inherited_attack_counts_in_budget', abs(float(out3[0, 5:].sum()) - .30) < 1e-6, out3[0, 5:])
neg = j.zeros((1, 8)).at[0, 6].set(-.05).at[0, 7].set(.1)
_, o = R.capped_payment(R.state(1), neg)
check('penalty_passes_and_does_not_consume_cap', abs(float(o[0, 6]) + .05) < 1e-6 and abs(float(o[0, 7]) - .1) < 1e-6, o[0])
# Stacking/empty bombs: THREAT/PRESSURE need explosion & reduced/cover & novel/near; CHAIN voids empty -> no source term pays placement.
check('threat_requires_effect', f"{THREAT}*(explosion&reduced&(novel&near)" in after_transition, 'THREAT gated by explosion, reduced escape, novel near cell')
check('pressure_requires_new_batch_and_limit', 'pressure_count' in after_transition and f'{PRESSURE}*pressure' in after_transition, 'pressure <=10, new batch')
# Release-bun / kill-slow-carrier / recapture loop: custody phi is a state function (closed loop = 0); attack credit
# is bounded per episode by the joint budget, so the variant adds at most .60-.45 = .15 per episode, never per loop.
from scripts.bun_native_v2_custody_reward import graded_phi
phi = lambda h, c, l: float(graded_phi(j.array(h), j.array(c), j.array(l), j.array(.5), j.array(.5)))
check('loop_custody_telescopes_0', abs((phi(False, True, False) - phi(True, False, False)) + (phi(True, False, False) - phi(False, True, False))) < 1e-9, 0)
s = R.state(1); total = 0.; last = None
for loop in range(20):
    s, o = R.capped_payment(s, j.zeros((1, 8)).at[0, 5:].set(.1)); last = float(o[0, 5:].sum()); total += last
check('loop_attack_credit_bounded', total <= ATTACK_BUDGET + 1e-6 and last == 0, dict(total=total, last=last))
atomic_json('runs/cc_train_20261105/training/league/ATK_REWARD_V2_TESTS.json',
            dict(status='PASS', version='cc-attack-reward-v2', threat=THREAT, pressure=PRESSURE, chain=CHAIN, caps=CAPS, attack_budget=ATTACK_BUDGET, tests=OUT, unix=time.time()))
print('PASS', len(OUT), 'tests')
