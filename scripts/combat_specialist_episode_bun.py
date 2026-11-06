"""Private bun collector with immutable episode-bound two-generation opponents.

Only the current generation argument changes on a 40-update refresh. Every
physical environment retains its selected generation until an actual reset.
The caller must assert that the bank generation to overwrite has no users.
"""
from types import ModuleType


def build(parent,accepted,*,include_c_reward=False):
    source=parent.src
    old='    return jax.pmap(collect,in_axes=(0,0,0,0,None,None,0,0))'
    assert source.count(old)==1
    source=source.replace(old,'    return collect')
    source=source.replace('def collect(p,st,ax,keys,bs,ba,ring,fixed):',
        'def collect(p,st,ax,keys,bs,ba,ring,fixed,current,current_generation):')
    source=source.replace('def choose_with_diagnostics(p,ring,fixed,states,aux,keys,groups):',
        'def choose_with_diagnostics(p,ring,fixed,states,aux,keys,groups,current):')
    source=source.replace('choose_with_diagnostics(p,ring,fixed,st,ax,key,groups)',
        'choose_with_diagnostics(p,ring,fixed,st,ax,key,groups,current)')
    begin=source.index("    for name,actor in [('current',p['actor']),('ring',ring),('fixed',fixed)]:")
    finish=source.index('    return nk,ke,joint_actions',begin)
    original=source[begin:finish]
    lines=original.splitlines()
    lines[0]="    for name,bank in [('current',current),('ring',ring),('fixed',fixed)]:"
    point=next(i for i,line in enumerate(lines) if 'mv,ab,_,_=E.train.net_forward' in line)
    lines[point:point+1]=[
        '        generation=aux[\'combat_mix\'][\'opponent_generation\'][ids]',
        '        logits=[]',
        '        for bank_generation in range(2):',
        '            actor=jax.tree.map(lambda x:x[bank_generation],bank)',
        "            mv,ab,_,_=E.train.net_forward(dict(actor=actor,critic=p['critic']),'transformer',obs,gv)",
        '            logits.append((mv,ab))',
        '        mv=j.where(generation[:,None]==0,logits[0][0],logits[1][0])',
        '        ab=j.where(generation[:,None]==0,logits[0][1],logits[1][1])']
    source=source[:begin]+'\n'.join(lines)+'\n'+source[finish:]
    anchor="            fa=dict(fa,combat_mix=metadata(fst,kinds,designated,seats,fa['r1_attribute_highwater']))"
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n            fa=dict(fa,combat_mix=dict(fa['combat_mix'],opponent_generation=j.full(count,current_generation,j.int32)))")
    anchor='            diagnostics.update(policy_diag)'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n            diagnostics.update(opponent_generation_ticks=jax.nn.one_hot(ax['combat_mix']['opponent_generation'],2).sum(0),opponent_generation_episodes=(jax.nn.one_hot(fa['combat_mix']['opponent_generation'],2)*done[:,None]).sum(0))"+(
        "\n            diagnostics.update({k:v.sum(0) for k,v in sm.items() if k.startswith('c_')})" if include_c_reward else ''))
    source=source.replace('output=(o,g,own,lp,v,nv,reward,done,mm,bm,diagnostics)',
        'output=(o,g,own,lp,v,nv,reward,done,mm,bm,live&st.alive[j.arange(count),seat],diagnostics)')
    source=source.replace('o,g,act,lp,v,nv,r,done,mm,bm,diag=d','o,g,act,lp,v,nv,r,done,mm,bm,live,diag=d')
    source=source.replace('(o,g,act,lp,adv,adv+v,mm,bm),','(o,g,act,lp,adv,adv+v,mm,bm,live),')
    impl=ModuleType('specialist_private_episode_bun')
    exec(compile(source,__file__,'exec'),impl.__dict__)
    for name in ('fresh','transition','metadata','inputs','prediction','E'):setattr(impl,name,getattr(accepted,name))
    impl.ATTACK=True
    impl._league_source=source
    return impl
