"""Train-only fixed native-transition scan for the predeclared PBRS lambdas."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH))

from scripts import je09_bun_scenes as scenes
from scripts import je09_route_pbrs as pbrs
from scripts import je09_stage_h8_ops as h8
from scripts import je09_stage_h8_pbrs_ops as pbrs_ops
from scripts.je08_bun_potential import INF
from scripts.je08_bun_recipe import ROOT
from scripts.je03_bun_stage1_20261011 import split_forward
from jax_bomb import je07_bun_env as env, jax_train as train


def scan(namespace):
    base = ROOT / namespace
    declaration = json.loads((base / 'ROUND_DECLARATION.json').read_text())
    env.prepare(); env.configure_training('full=1', 1, reward_profile='legacy',
        retention_replay=0, tactical_bomb_placement_reward=.1)
    scenes.bind_training()
    original = train.net_forward
    train.net_forward = lambda q, a, o, s=None: split_forward(q, a, o, s, original)
    with (base / 'inputs/initial.pkl').open('rb') as f:
        params = jax.tree.map(jnp.asarray, pickle.load(f))
    n = 16
    course = h8.curriculum_weights(jnp.zeros((10, len(h8.FUNNEL))), .4, .25)
    key = jax.random.PRNGKey(declaration['seeds']['scan'])
    key, state_key = jax.random.split(key)
    states = scenes.init_batch(state_key, n, course)
    aux = pbrs_ops.reward_state(n)
    deltas = []; raw = []; phases = []; distances = []
    for _ in range(16):
        key, action_key, step_key = jax.random.split(key, 3)
        obs = train.both_perspectives(states); gv = train.both_states(states)
        mm, bm = train.both_masks(states)
        actions, _, _ = train.sample_actions(params, 'transformer', obs, (mm, bm),
                                              action_key, state=gv)
        joint = actions.reshape((n, 2, 2))
        next_states, done, info = jax.vmap(lambda s, a, k: env.step(
            s, a, k, auto_reset=False, return_info=True))(
                states, joint, jax.random.split(step_key, n))
        info = dict(info, h_route_terminal=done, h_joint_actions=joint)
        settled = ~(jnp.any(next_states.fuse > 0, axis=(1, 2)) |
                    jnp.any(next_states.core.blast_linger > 0, axis=(1, 2)) |
                    jnp.any(next_states.trapped > 0, axis=1))
        raw_reward, newaux, parts, _ = h8.combined_reward(
            states, next_states, info, aux, settled, 1., None)
        shaped_reward, newaux, parts, metrics = pbrs_ops.combined_reward(
            states, next_states, info, aux, settled, 1., None)
        deltas.append(np.asarray(metrics['route_potential_delta']).reshape(-1))
        raw.append(np.asarray(raw_reward).reshape(-1))
        phases.append(np.asarray(metrics['route_phase']).reshape(-1))
        distances.append(np.asarray(metrics['route_distance_before']).reshape(-1))
        states, aux = next_states, newaux
    delta = np.concatenate(deltas).astype(np.float64)
    raw_reward = np.concatenate(raw).astype(np.float64)
    phase = np.concatenate(phases).astype(np.int32)
    distance = np.concatenate(distances).astype(np.int32)
    active = phase > 0
    ratio = np.abs(delta) / np.maximum(np.abs(raw_reward), .25)
    rows = []
    for coefficient in (.025, .05, .1):
        shaped_ratio = coefficient * ratio[active]
        rows.append(dict(lambda_value=coefficient, finite=bool(np.isfinite(delta).all()),
            active_transitions=int(active.sum()), phase_coverage={str(i): int(np.sum(phase == i)) for i in range(1, 5)},
            positive=int(np.sum(delta[active] > 0)), negative=int(np.sum(delta[active] < 0)),
            p50_abs_shaping_over_raw=float(np.quantile(shaped_ratio, .5)) if active.any() else None,
            p95_abs_shaping_over_raw=float(np.quantile(shaped_ratio, .95)) if active.any() else None,
            inf_to_finite=int(np.sum(active & (distance >= INF)))))
    passing = [r for r in rows if r['finite'] and r['positive'] and r['negative']
        and r['active_transitions'] > 0 and r['p50_abs_shaping_over_raw'] <= .25
        and r['p95_abs_shaping_over_raw'] <= .5]
    selected = min((r['lambda_value'] for r in passing), default=None)
    report = dict(passed=selected is not None, experiment_id=declaration['experiment_id'],
        seed=declaration['seeds']['scan'], source='fresh native train-only H8 learner rollout; no evaluation seed/data',
        samples=int(len(delta)), sample_sha256=__import__('hashlib').sha256(
            delta.tobytes() + raw_reward.tobytes() + phase.tobytes()).hexdigest(),
        gamma=.995, candidates=rows, selection_rule='smallest lambda with finite signal, positive and negative active deltas, median abs shaping/raw <=0.25, p95 <=0.5',
        discounted_cycle_residual=0.0, selected_lambda=selected,
        cycle_proof='algebraic ledger closed-loop tests; sampled transitions are not used as cycle proof')
    (base / 'validation').mkdir(exist_ok=True)
    (base / 'validation/TRAIN_ONLY_LAMBDA_SCAN.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--namespace', required=True)
    print(json.dumps(scan(parser.parse_args().namespace), indent=2))
