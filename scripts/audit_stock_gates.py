"""Read-only production phase/teacher gates on fixed Web acquisition frames.

No reward history is invented: PBRS ledger residuals are reported separately.
The adapter reconstructs observable geometry, inventory and native windows;
empty-hand eligibility is also challenged with stale cached authority.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax_bomb import je07_bun_env as env
from scripts import je10_ops as ops,je10_route_oracle as oracle
from scripts.je08_bun_potential import shortest


def frame_state(frame,template):
    core_map={'pos':'pos','wall':'wall','brick':'brick','pushable':'pushable',
        'fuse':'fuse','blast_linger':'blastLinger','alive':'alive','t':'t'}
    core=template.core._replace(**{k:jnp.asarray(frame[v],getattr(template.core,k).dtype).reshape(getattr(template.core,k).shape) for k,v in core_map.items()})
    mapping={'bun_stored':'bunStored','bun_carried':'bunCarried','bun_loose':'bunLoose',
        'bun_respawn':'bunRespawn','trapped':'trapped'}
    return template._replace(core=core,scene=jnp.int32(8),**{
        k:jnp.asarray(frame[v],getattr(template,k).dtype).reshape(getattr(template,k).shape) for k,v in mapping.items()})


def audit(folder):
    env.prepare()
    template=env.init_state(jax.random.PRNGKey(0)) if hasattr(env,'init_state') else jax.tree.map(lambda x:x[0],env.init_batch(jax.random.PRNGKey(0),1))
    events=[];states=[]
    for path in sorted(Path(folder).glob('g*_*.json')):
        for ep in json.loads(path.read_text())['episodes']:
            for e in ep['acquisitions']:
                events.append(dict(generation=int(path.name[1:7]),seed=ep['seed'],seat=ep['team'],tick=e['tick_before'],window=e['window'],source=e['source']))
                states.append(frame_state(e['state_before'],template))
    s=jax.tree.map(lambda *xs:jnp.stack(xs),*states);n=len(events)
    aux=ops.reward_state(n)
    blocked=s.wall|s.brick|s.core.pushable
    bases=jnp.stack([env._base_mask(0),env._base_mask(1)])
    goals=jnp.concatenate([jnp.zeros((n,2,13,15),bool),jnp.broadcast_to(bases[::-1],(n,2,13,15)),jnp.broadcast_to(bases,(n,2,13,15))],1)
    distance=jax.jit(shortest)(blocked[:,None],jnp.zeros_like(blocked[:,None]),goals)
    route=dict(aux['routing'],stock_distance=distance[:,2:4],distance=jnp.concatenate([distance[:,:2],distance[:,4:]],1))
    aux['dagger'].update(window=jnp.ones((n,2),bool),custody=jnp.ones((n,2),bool),blocked=blocked,distance=distance)
    phase=ops.safe_phase_values(s,route,aux['drop'])
    seat=jnp.array([e['seat'] for e in events]);rows=jnp.arange(n)
    labels=jax.jit(oracle.labels)(s,aux,jnp.ones((n,5),bool),seat,jnp.zeros((n,13,15)))
    from scripts.prestock_training_state import state,update,potential
    aux['routing']=route
    selected=jax.jit(update)(s,aux,state(n));values=potential(s,selected,phase)
    for i,e in enumerate(events):
        who=e['seat'];e.update(phase=int(phase[0][i,who]),prepared_phase=int(values[0][i,who]),
            preparation_active=bool(selected['active'][i,who]),dagger_candidate_kind=int(labels[i,6]),
            dagger_active=bool(labels[i,7]),dagger_label_accepted=bool(labels[i,5]))
        if e['window']=='FREE_ALIVE':
            assert e['phase']==e['prepared_phase']==0 and not e['preparation_active'] and not e['dagger_active'] and not e['dagger_label_accepted'],e
    out=dict(events=events,free_stock_events=sum(e['window']=='FREE_ALIVE' and e['source']=='STOCK' for e in events),
        free_stock_phase_or_teacher_violations=0,scope='Production current native gates on recorded before states; geometry rebuilt; stale cached window/custody=True challenge. No claim about unrecorded PPO reward history.')
    (Path(folder)/'PRODUCTION_GATES.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='events'},indent=2))


if __name__=='__main__':
    import sys
    audit(sys.argv[1])
