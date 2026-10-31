"""Scheduling-only contracts; no model, reward or action changes."""
import hashlib,json


def chunk_updates(config):
    value=int(config.get('chunk_updates',320))
    if value<320 or value%40:raise ValueError('chunk must be >=320 and divisible by 40')
    return value


def should_evaluate(update):return int(update)%40==0


def prewarm_fingerprint(protocol):
    config=dict(common=protocol['common'],recipe=next(iter(protocol['arms'].values())),
        source_sha256=protocol['source_sha256'],spawn_manifest=protocol['spawn_manifest'],
        rule_version=protocol['rule_version'],reward_version=protocol['reward_version'],slot=protocol['slot'])
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def valid_prewarm(smoke,restore):
    metric=smoke.get('checkpoint',{}).get('metrics',{})
    return bool(smoke.get('passed') and metric.get('finite') and
        all(metric.get('update_l2',{}).get(k,0)>0 for k in ('actor','critic')) and
        all(restore.get(k) is True for k in ('passed','fresh_process','full_env_aux_RNG_exact','next_step_exact','Adam_exact','ring_exact')))
