"""Native-JAX / production-JS both-seat inputs and logits, matched state only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                  JAXBOMB_RULE='bun',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,pickle,subprocess,time
from pathlib import Path
import jax,jax.numpy as j,numpy as np
from scripts.population_run_state import atomic_json,sha
from scripts import combat_item_engine as T,carrier_mechanism_engine as E
from scripts.checkpoint_policy_artifact import load_actor

OUT=Path('runs/bomb_threat_v1/iter2/seat_asym_audit')


def exchange_encoder_identities(states):
    """Diagnostic-only identity relabeling, with unchanged spatial coordinates."""
    core_fields=('pos','alive','hp','invuln','bombs_cap','blast_cap','spd_g','buffs','debuffs','items')
    core=states.core._replace(**{k:getattr(states.core,k)[:,::-1] for k in core_fields},
        owner=j.where(states.core.owner>=0,1-states.core.owner,states.core.owner))
    per_player=('bun_respawn','bun_spawn_pos','held_item','move_status','status_ticks',
                'slide_dir','last_move_dir','item_slots','item_counts','trapped')
    return states._replace(core=core,**{k:getattr(states,k)[:,::-1] for k in per_player},
        bun_carried=j.where(states.bun_carried[:,::-1]>=0,1-states.bun_carried[:,::-1],-1),
        bun_stored=states.bun_stored[:,::-1,::-1],bun_loose=states.bun_loose[:,:,:,::-1],
        field_owner=j.where(states.field_owner>=0,1-states.field_owner,states.field_owner))


def main():
    OUT.mkdir(parents=True,exist_ok=True);T.configure()
    model='runs/bomb_threat_v1/eval/bt_slot01_c000015_u000720/model/model.json'
    cp=Path('runs/bomb_threat_v1/formal/bt_slot01_c000015/u000720');actor=load_actor(cp)
    states,_=jax.jit(lambda keys:E.initial(keys,0))(jax.random.split(jax.random.PRNGKey(2026102917),4))
    # Diagnostic encoder fixture only: controlled flags exercise relative channels,
    # never enters a collector or training Bank and is not an acquisition sample.
    core=states.core._replace(pos=j.array([[[4.5,2.5],[8.5,12.5]],[[3.5,1.5],[9.5,13.5]],
        [[5.5,2.5],[7.5,12.5]],[[2.5,1.5],[10.5,13.5]]]),t=j.array([0,40,80,120],j.int32),
        alive=j.array([[1,1],[1,1],[1,0],[1,1]],bool),
        hp=j.array([[1,2],[2,1],[1,0],[2,2]],j.int32),
        fuse=states.fuse.at[:,4,4].set(80).at[:,8,10].set(70),
        owner=states.core.owner.at[:,4,4].set(0).at[:,8,10].set(1))
    states=states._replace(core=core,trapped=j.array([[0,0],[12,0],[0,0],[0,24]],states.trapped.dtype),
        bun_carried=j.array([[-1,-1],[1,-1],[-1,0],[0,1]],states.bun_carried.dtype),
        bun_stored=j.array([[[1,0],[0,1]],[[1,0],[0,0]],[[0,0],[0,1]],[[0,0],[0,0]]],states.bun_stored.dtype),
        bun_loose=states.bun_loose.at[:,3,4,0].set(1).at[:,9,10,1].set(1),
        field_owner=states.field_owner.at[:,3,5].set(0).at[:,9,9].set(1),
        field_armed=states.field_armed.at[:,3,5].set(True).at[:,9,9].set(True))
    o=E.train.both_perspectives(states);g=E.train.both_states(states)
    mv,ab,_,_=E.train.net_forward(actor,'transformer',o,g);jax.block_until_ready(mv)
    exchanged=exchange_encoder_identities(states)
    original_mask=E.env._base_mask
    try:
        E.env._base_mask=lambda team:original_mask(1-team)
        so=E.train.both_perspectives(exchanged);sg=E.train.both_states(exchanged)
        sm,sa,_,_=E.train.net_forward(actor,'transformer',so,sg)
        order=j.array([4,5,6,7,0,1,2,3]);jax.block_until_ready(sm)
        identity_swap={k:float(np.max(np.abs(np.asarray(a)-np.asarray(b[order]))))
            for k,a,b in [('obs_max',o,so),('state_max',g,sg),('move_max',mv,sm),('ability_max',ab,sa)]}
    finally:E.env._base_mask=original_mask
    from scripts.je03_native_vit_export_20261011 import NativeViT
    import torch
    torch.set_num_threads(1)
    with torch.no_grad():fp_mv,fp_ab,_=NativeViT(actor)(torch.from_numpy(np.asarray(o).copy()),torch.from_numpy(np.asarray(g).copy()))
    payload=[]
    fields={'pos':'pos','alive':'alive','hp':'hp','invuln':'invuln','bombsCap':'bombs_cap',
        'blastCap':'blast_cap','spdG':'spd_g','wall':'wall','brick':'brick','fuse':'fuse',
        'owner':'owner','bombBlast':'bomb_blast','blastLinger':'blast_linger','bush':'bush','crate':'crate',
        'pushable':'pushable','trapped':'trapped','bunCarried':'bun_carried','bunStored':'bun_stored',
        'bunLoose':'bun_loose','bunRespawn':'bun_respawn','heldItem':'held_item','movementStatus':'move_status',
        'fieldItem':'field_item','fieldOwner':'field_owner','fieldArmed':'field_armed','t':'t'}
    for index in range(4):
        st=jax.tree.map(lambda x:np.asarray(x[index]),states)
        get=lambda key:getattr(st,key) if hasattr(st,key) else getattr(st.core,key)
        js={key:get(value).tolist() for key,value in fields.items()}
        payload.append(dict(id=index,seed=2026102917+index,js=js,
            crate=np.asarray(st.core.crate).ravel().tolist(),tactical_crate=np.asarray(st.tactical_crate).ravel().tolist()))
    atomic_json(OUT/'ENCODING_INPUT.json',dict(model=model,rows=payload))
    subprocess.run(['node','scripts/bomb_threat_seat_encoding.js',str(OUT/'ENCODING_INPUT.json'),str(OUT/'ENCODING_JS.json')],check=True)
    js=json.loads((OUT/'ENCODING_JS.json').read_text());result=[]
    from jax_bomb import je05_core_env as base
    native_directions=[]
    for direction in js['directions']:
        p=j.array([5.5,6.5]) if direction['seat']==0 else j.array([7.5,8.5])
        after=base._move_player(p,j.array(direction['move']),j.array(True),j.zeros((13,15),bool))
        delta=np.asarray(after-p)
        assert np.array_equal(np.sign(delta),direction['expected'])
        native_directions.append(dict(seat=direction['seat'],move=direction['move'],delta=delta.tolist()))
    for index,row in enumerate(js['rows']):
        for seat in range(2):
            idx=index+seat*4;target=row['original'][seat]
            obsdiff=np.abs(np.asarray(o[idx])-np.asarray(target['obs']).reshape(24,13,15));gd=np.abs(np.asarray(g[idx])-target['state'])
            result.append(dict(fixture=index,seat=seat,obs_max=float(obsdiff.max()),
                per_channel_max=obsdiff.max((1,2)).tolist(),state_max=float(gd.max()),state_differences=np.flatnonzero(gd>2e-5).tolist(),
                move_logit_max=float(np.max(np.abs(np.asarray(mv[idx])-target['move']))),
                ability_logit_max=float(np.max(np.abs(np.asarray(ab[idx])-target['ability']))),
                fp32_export_move_max=float(np.max(np.abs(fp_mv[idx].numpy()-target['move']))),
                fp32_export_ability_max=float(np.max(np.abs(fp_ab[idx].numpy()-target['ability']))),
                native_bf16_move_argmax=int(np.argmax(np.asarray(mv[idx]))),js_fp32_move_argmax=int(np.argmax(target['move'])),
                js_identity_swap=row['relabeled'][seat]))
    status='PASS_INPUT_MAPPING_WITH_DISCLOSED_PRECISION_DIFFERENCE' if max(identity_swap.values())<2e-5 and all(x['obs_max']<=2e-5 and x['state_max']<=2e-5 and x['fp32_export_move_max']<.002 and x['fp32_export_ability_max']<.002 and max(x['js_identity_swap'].values())<2e-5 for x in result) else 'MISMATCH_REQUIRES_REVIEW'
    atomic_json(OUT/'ENCODING_RESULT.json',dict(status=status,rows=result,policy_sha256=sha(cp/'policy.pt'),model_sha256=sha(model),
        actor_uses_absolute_coordinates=True,action_direction_remap='identity: 0up 1down 2left 3right 4stay',
        native_forward_precision='jax_net._transformer_pool/_tf_block bfloat16; not fp32 export',
        jax_identity_swap=identity_swap,js_action_direction=js.get('directions'),jax_action_direction=native_directions,
        fixtures=4,synthetic_encoder_only=True,training_eligible=False,unix=time.time()))
    print(json.dumps(dict(status=status,rows=result)))


if __name__=='__main__':main()
