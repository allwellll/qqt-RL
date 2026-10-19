"""Host-only provenance of the actual bounded payload passed to DAgger loss."""
import hashlib
import numpy as np


def summarize(replay,generation,source_actor_digest):
    """Hash selected rows in slot order, preserving replacement duplicates.

    This reads the small selected replay, never the rollout or environment.
    Slot halves describe the actual current/history draws, even on fallback.
    """
    data={k:np.asarray(replay[k]) for k in ['obs','gv','legal','ability_legal',
        'joint_target','valid','kind','preparation','opening','round']}
    valid=data['valid'];devices,size=valid.shape
    assert size%2==0 and all(v.shape[:2]==valid.shape for v in data.values())
    rows=[];counts={}
    for half,begin,end in [('current',0,size//2),('history',size//2,size)]:
        v=valid[:,begin:end];prep=data['preparation'][:,begin:end]
        opening=data['opening'][:,begin:end];kind=data['kind'][:,begin:end]
        counts[half]=dict(valid=int(v.sum()),returning=int((v&(kind==3)&~prep).sum()),
            navigation=int((v&prep&~opening).sum()),opening=int((v&opening).sum()))
    for device in range(devices):
        for slot in range(size):
            if not valid[device,slot]:continue
            observation=hashlib.sha256(data['obs'][device,slot].tobytes()
                +data['gv'][device,slot].tobytes()).hexdigest()
            label=hashlib.sha256()
            for name in ['legal','ability_legal','joint_target']:
                a=data[name][device,slot]
                label.update(name.encode());label.update(str(a.shape).encode())
                label.update(str(a.dtype).encode());label.update(a.tobytes())
            rows.append(dict(device=device,slot=slot,half='current' if slot<size//2 else 'history',
                query_generation=int(data['round'][device,slot]),kind=int(data['kind'][device,slot]),
                preparation=bool(data['preparation'][device,slot]),opening=bool(data['opening'][device,slot]),
                observation_sha256=observation,label_sha256=label.hexdigest()))
    return dict(schema='je10.actual_dagger_replay_exposure/v1',generation=int(generation),
        source_actor_digest=source_actor_digest,devices=devices,slots_per_device=size,
        rows=rows,counts=counts,controls_training=False,optimizer_steps=0,
        scope='Actual selected bounded DAgger payload before optimizer reuse; duplicates retained, not unique-state counts or every microbatch reuse.',
        observation_hash_scope='Same raw obs+global bytes as learner first-deviation trace, not full native state.',
        source_actor_scope='Raw Actor used to collect and select this generation, before its PPO update.')
