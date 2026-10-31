"""Read-only same-state phase counterfactual; never a capability endpoint."""
import argparse
import json
import os
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from scripts import retention_phase_conditioning as phase
from scripts.population_run_state import atomic_json, sha


def audit(checkpoint, sample, target):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and jax.default_backend() == 'cpu'
    checkpoint, sample, target = map(Path, [checkpoint, sample, target])
    with checkpoint.open('rb') as stream:
        raw = pickle.load(stream)
    pair = raw['live']['params'] if 'live' in raw else raw
    if 'live' in raw:
        pair = jax.tree.map(lambda x: x[0], pair)
    pair = pair if 'actor' in pair else dict(actor=pair)
    data = np.load(sample)
    obs, gv = jnp.asarray(data['obs']), jnp.asarray(data['gv'])
    expected = np.asarray(phase.phase_id(obs, gv))
    records = {}
    for role, params in pair.items():
        params = jax.tree.map(jnp.asarray, params)
        baseline = phase.forward(params, obs, gv)
        move, ability = [jax.nn.softmax(x, -1) for x in baseline[:2]]
        counterfactuals = []
        for index in range(5):
            encoded = jnp.broadcast_to(jax.nn.one_hot(index, 5), (len(obs), 5))
            cf = phase.forward(params, obs, gv, feature_override=encoded)
            cm, ca = [jax.nn.softmax(x, -1) for x in cf[:2]]
            kl = (move*(jnp.log(jnp.maximum(move,1e-9))-jnp.log(jnp.maximum(cm,1e-9)))).sum(-1)
            kl += (ability*(jnp.log(jnp.maximum(ability,1e-9))-jnp.log(jnp.maximum(ca,1e-9)))).sum(-1)
            counterfactuals.append(dict(forced_phase=index,
                move_probability_l1=float(jnp.abs(cm-move).sum(-1).mean()),
                ability_probability_l1=float(jnp.abs(ca-ability).sum(-1).mean()),
                policy_kl=float(kl.mean()),value_abs_delta=float(jnp.abs(cf[2]-baseline[2]).mean()),
                move_changed=int((cf[0].argmax(-1)!=baseline[0].argmax(-1)).sum())))
        slices = {}
        for label, subset in [('all_diagnostic', np.ones(len(obs),bool)),
                              ('natural_complete', data['task']=='natural_full2400'),
                              ('artificial_carry', data['artificial_carry']),
                              ('other_route_diagnostic', (data['task']!='natural_full2400')&~data['artificial_carry'])]:
            by_phase = {}
            for index in range(5):
                mask = (expected == index)&subset
                by_phase[str(index)] = dict(count=int(mask.sum()),
                    move_probability_mean=np.asarray(move)[mask].mean(0).tolist() if mask.any() else None,
                    ability_probability_mean=np.asarray(ability)[mask].mean(0).tolist() if mask.any() else None)
            slices[label]=by_phase
        records[role] = dict(conditioned=params['state_w'].shape[-2]==29,
                             phase_action_distributions_by_source=slices,counterfactuals=counterfactuals)
    result = dict(passed=True,checkpoint_sha256=sha(checkpoint),sample_sha256=sha(sample),
        states=len(obs),networks=records,diagnostic_only=True,controls_training=False,
        limitation='Fixed same visible states; forced phases are counterfactual and may be inconsistent. '
                   'Different action distributions by actual phase are confounded by different states; '
                   'only same-state phase intervention isolates feature usage. No phase prediction head. '
                   'Probe includes separately labelled artificial carry diagnostics, never training data.')
    atomic_json(target,result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('sample');p.add_argument('target')
    a=p.parse_args();audit(a.checkpoint,a.sample,a.target)
