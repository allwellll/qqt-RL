"""Prepared terminal-only shared pre-stock objective/joint-supervision wiring.

Building patches is read-only. Installation requires a complete fixed parent
and an eight-device release receipt; no capability threshold is consulted.
"""
import ast
import json
from pathlib import Path


def replace_once(source,old,new):
    assert source.count(old)==1,'unrecognized candidate integration anchor'
    return source.replace(old,new,1)


def patched_sources(sources):
    result=dict(sources)
    name='scripts/je10_ops.py';s=result[name]
    if '# Shared pre-stock auxiliary state.' not in s:
        s=replace_once(s,"    aux['route_pbrs']=pbrs.state(n)","""    aux['route_pbrs']=pbrs.state(n)
    # Shared pre-stock auxiliary state.
    from scripts.prestock_training_state import state as prestock_state
    aux['prestock']=prestock_state(n)""")
        s=replace_once(s,'    dagger.COEFFICIENT=.2',"""    dagger.COEFFICIENT=.2
    from scripts import prestock_joint_replay as joint_replay
    dagger.buffer=joint_replay.buffer
    dagger.aggregate=joint_replay.aggregate
    dagger.replay_loss=joint_replay.replay_loss""")
    result[name]=s
    name='scripts/je09_stage_h5c_ops.py';s=result[name]
    if '# Explicit shared preparation selection.' not in s:
        s=replace_once(s,'        states,aux,key,kinds,counts,poolcounts,component=carry',"""        states,aux,key,kinds,counts,poolcounts,component=carry
        # Explicit shared preparation selection.
        from scripts.prestock_training_state import update
        aux=dict(aux,prestock=update(states,aux,aux['prestock']))""")
        s=replace_once(s,'        context=jnp.concatenate([context,dagger.labels(states,aux,mm[idx],seat,obs[idx,5])],-1)',"""        route_labels=dagger.labels(states,aux,mm[idx],seat,obs[idx,5])
        context=jnp.concatenate([context,route_labels[:, :12]],-1)""")
        anchor='        data=(obs[idx],gv[idx],acts[idx],lps[idx],vals[idx],rewards[rows,seat],done,(mm[idx],bm[idx]),context)'
        s=replace_once(s,anchor,"        assert context.shape[-1]==95,'prestock requires the declared return/chase/trace layout'\n"+'        context=jnp.concatenate([context,route_labels[:, 12:]],-1)\n'+anchor)
    result[name]=s
    name='scripts/je09_stage_h6_ops.py';s=result[name]
    if '# Explicit bounded joint teacher.' not in s:
        tree=ast.parse(s);collect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect')
        node=next(n for n in collect.body if isinstance(n,ast.FunctionDef) and n.name=='verified_labels')
        lines=s.splitlines(keepends=True)
        replacement='''    def verified_labels(states, aux, legal, seat, danger):
        # Explicit bounded joint teacher.
        from scripts.prestock_verified_labels import verified_labels as verify
        return verify(states,aux,legal,seat,danger,original_labels,slots)
'''
        s=''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:])
        old='''    objective_selected = jnp.take_along_axis(proposed, jnp.clip(move, 0, 4)[..., None], axis=-1)[..., 0] > 0
    fragment = earliest_fragment(c)
    mismatch = objective_active & (d[..., 5] > 0) & ~objective_selected & (fragment > 0)'''
        new='''    from scripts.prestock_fragment_queries import joint_mismatch,first_deviations
    joint=c[...,95:110].reshape(c.shape[:-1]+(5,3))>0
    fragment = earliest_fragment(c)
    mismatch=joint_mismatch(joint,actions,objective_active&(d[...,5]>0))&(fragment>0)'''
        s=replace_once(s,old,new)
        s=replace_once(s,'    first = first_fragment_deviations(mismatch, fragment, done)',
            '    first = first_deviations(mismatch,fragment,done,c[...,110]>0,c[...,111].astype(jnp.int32),c[...,113]>0)')
    result[name]=s
    name='scripts/je09_route_pbrs.py';s=result[name]
    if '# Shared native preparation potential.' not in s:
        now='    phase, identity, base_now, dist_now, provenance = phase_values(states, now_route, aux[\'drop\'])'
        s=replace_once(s,now,now+'''
    # Shared native preparation potential.
    from scripts.prestock_training_state import potential as prestock_potential,update as prestock_update
    phase,identity,base_now,dist_now,provenance=prestock_potential(states,aux['prestock'],
        (phase,identity,base_now,dist_now,provenance))''')
        nxt='''    phase_next, identity_next, base_next, dist_next, _ = phase_values(
        next_states, dict(new_route, stock_distance=stock_next), newaux['drop'])'''
        s=replace_once(s,nxt,nxt+'''
    next_selection_aux=dict(newaux,terminal=aux['terminal']|info['h_route_terminal'])
    newaux['prestock']=prestock_update(next_states,next_selection_aux,aux['prestock'])
    phase_next,identity_next,base_next,dist_next,_=prestock_potential(next_states,newaux['prestock'],
        (phase_next,identity_next,base_next,dist_next,jnp.zeros_like(identity_next)))''')
    result[name]=s
    name='scripts/je10_worker.py';s=result[name]
    if '# Joint preparation provenance.' not in s:
        old='''            chosen=np.take_along_axis(oracle_target,np.asarray(batch[2])[...,0:1],axis=-1)[...,0]
            mismatch=(ctx[...,55]>0)&(oracle_target.sum(-1)>0)&(chosen<=0)'''
        new='''            # Joint preparation provenance.
            actual=np.asarray(batch[2]);pair=actual[...,0]*3+actual[...,1]
            joint_oracle=ctx[...,95:110].reshape(ctx.shape[:-1]+(5,3))>0
            chosen=np.take_along_axis(joint_oracle.reshape(ctx.shape[:-1]+(15,)),pair[...,None],axis=-1)[...,0]
            mismatch=(ctx[...,55]>0)&joint_oracle.any((-2,-1))&~chosen
            preparing=ctx[...,110]>0;opening=ctx[...,113]>0
            row['dagger']['prestock']=dict(active=int(preparing.sum()),selected=int((preparing&(ctx[...,56]>0)).sum()),
                accepted=int((preparing&joint_oracle.any((-2,-1))).sum()),query=int((preparing&(ctx[...,53]>0)).sum()),
                navigation_accepted=int((preparing&joint_oracle[:,:, :, :,0].any(-1)).sum()),
                opening_accepted=int(opening.sum()),opening_query=int((opening&(ctx[...,53]>0)).sum()),
                scope='All native route scenes, bounded selected slots; per-rollout episode/revision quotas.')'''
        # Avoid assuming a particular number of leading rollout dimensions.
        new=new.replace('joint_oracle[:,:, :, :,0]','joint_oracle[...,0]')
        s=replace_once(s,old,new)
        anchor='                    phase=int(ctx[dev,tick,env_index,54]),trace='
        s=replace_once(s,anchor,"                    joint_oracle=joint_oracle[dev,tick,env_index].astype(int).tolist(),preparation_metadata=ctx[dev,tick,env_index,110:114].tolist(),\n"+anchor)
    result[name]=s
    name='scripts/je10_campaign.py'
    if name in result:
        s=result[name]
        if "worker.STATE_SCHEMA=read(Path(run)/'plan.json')['state_schema']" not in s:
            s=replace_once(s,"worker.STATE_SCHEMA='je10-control-route-v38'", "worker.STATE_SCHEMA=read(Path(run)/'plan.json')['state_schema']")
        result[name]=s
    name='scripts/terminal_checkpoint_restore.py'
    if name in result and "verify_state_schema(proof,json.loads((base/'plan.json').read_text()))" not in result[name]:
        result[name]=replace_once(result[name],"    assert proof['state_schema']=='je10-control-route-v38'",
            "    verify_state_schema(proof,json.loads((base/'plan.json').read_text()))")
    name='scripts/je10_evaluate.py'
    if name in result and '# Native joint fit under the same CPU authority.' not in result[name]:
        result[name]=replace_once(result[name],'    return result\n', '''    # Native joint fit under the same CPU authority.
    # Fixed quick-screen COMPLETE is already durable; audit is diagnostic only.
    if read(BASE/'plan.json').get('prestock_preparation'):
        try:
            from scripts.audit_joint_dagger import audit as joint_fifo_audit
            joint_fifo_audit(BASE,g)
        except Exception as error:
            atomic_json(out/'DAGGER_JOINT_AUDIT_FAILURE.json',dict(error=str(error),controls_gpu=False,
                fixed_evaluation_complete=True,capability_gate=False))
    return result
''')
    for s in result.values():ast.parse(s)
    return result


def install(root,parent):
    root=Path(root);parent=Path(parent)
    terminal=json.loads((parent.parent/'TRAINER_EXIT.json').read_text())
    assert terminal['generation']==128 and terminal['passed'] and not terminal['atomic_stop']
    release=json.loads((parent.parents[2]/'GPU_RELEASE_RECEIPT.json').read_text())
    assert release['released'] and release['compute_clients']==0
    files=['scripts/je10_ops.py','scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h6_ops.py',
        'scripts/je09_route_pbrs.py','scripts/je10_worker.py','scripts/je10_campaign.py',
        'scripts/terminal_checkpoint_restore.py','scripts/je10_evaluate.py']
    previous={name:(root/name).read_text() for name in files}
    patched=patched_sources(previous)
    for name,source in patched.items():
        if source!=previous[name]:(root/name).write_text(source)
    return dict(changed_factor='shared pre-stock corridor objective with native-safe joint learner-state supervision',
        new_auxiliary_state=True,actor_observation_changed=False,reward_coefficient_changed=False,
        query_slots_changed=False,ppo_changed=False,scene_mixture_changed=False,
        extra_teacher_steps='Restricted forty-step opening witness, scalar conditional per bounded stock slot',
        causal_scope='One objective mechanism integrating PBRS phase and joint DAgger; sequential followup, no same-parent control')
