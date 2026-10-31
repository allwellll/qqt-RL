"""Opt-in R4 curriculum; original R3 reward and runtime stay reproducible."""
import ast
from types import ModuleType


def install(module, fraction=.30):
    if getattr(module, '_r4_installed', False):
        return
    if fraction != .30:
        raise ValueError('R4 K1/K3 freeze thirty percent combat curriculum')
    # Both construction and every reset call the very same fresh function.
    original = module.fresh
    from scripts import bun_native_v2_custody_collector as custody
    tree = ast.parse(custody.source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fresh')
    source = ast.get_source_segment(custody.source, node)
    assert source.count('.25/.9') == 1
    source = source.replace('.25/.9', '.25/.7')
    anchor = '    ar,aa=E.initial(keys,1)'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + '''
    # Genuine empty-hand clear-brick states, never fabricated carriers.
    lesson=jax.vmap(lambda k:jax.random.bernoulli(jax.random.fold_in(k,0xC047)))(keys)
    ar=ar._replace(lesson=j.where(lesson,E.env.LESSON_COMBAT_KILL,E.env.LESSON_KILL_RUSH))''')
    scope = dict(original.__globals__)
    exec(compile(source, __file__, 'exec'), scope)
    fresh = scope['fresh']
    source = module.src
    anchor = '            diagnostics.update(policy_diag)'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + '''
            diagnostics['r4_lesson_ticks']=j.stack([(st.lesson==v).sum() for v in (0,7,12)])
            diagnostics['r4_lesson_resets']=j.stack([(done&(fst.lesson==v)).sum() for v in (0,7,12)])
            diagnostics['r4_curriculum_ticks']=(st.lesson==7).sum()+(st.lesson==12).sum()''')
    impl = ModuleType('r4_private_collector')
    exec(compile(source, __file__, 'exec'), impl.__dict__)
    for name in ('transition', 'metadata', 'inputs', 'prediction'):
        setattr(impl, name, getattr(module._impl, name))
    impl.fresh = fresh
    module.fresh = fresh
    module._impl = module._base._impl = impl
    for name in ('choose', 'choose_with_diagnostics'):
        setattr(module, name, getattr(impl, name))
    module._r4_installed = True
