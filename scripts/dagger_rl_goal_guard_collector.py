"""Inactive collector adapter: only goal qualification plus passive counts."""
import inspect
import numpy as np


def summary(flags, sources, evidence):
    flags, sources, evidence = map(np.asarray, [flags, sources, evidence])
    if sources.shape != flags.shape[:-1] or evidence.shape != sources.shape+(3,):
        raise ValueError('unaligned goal-guard telemetry')
    if not np.isfinite(sources).all():
        raise ValueError('invalid source metadata')
    rows = {}
    for name, source in [('natural', sources < 0), ('real_pickup_bank', sources >= 0)]:
        for phase, present in [('carry', flags[..., 0] > 0),
                               ('empty_window', flags[..., 15] > 0),
                               ('empty_combat', flags[..., 14] > 0)]:
            selected = source & present
            values = evidence[selected].sum(0).astype(int)
            rows[name+'/'+phase] = dict(ticks=int(selected.sum()),
                legacy_route_rows=int(values[0]), removed_goal0_route_rows=int(values[1]),
                removed_final_teacher_rows_after_item_priority=int(values[2]))
    return dict(rows=rows, total_ticks=int(sources.size),
                scope='Learner query ticks; final-label removal respects item priority. '
                      'PPO-stage queries are not adopted DAgger; retained replay exposure is separate.')


def factory(template, steps, *, legacy_teacher, stream, counters):
    source = inspect.getsource(template.collector)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('frozen collector anchor changed: '+old)
        source = source.replace(old, new)

    replace('            route = C.T.route_teacher(st, ax)',
            '            legacy_route = legacy_teacher(st, ax)\n'
            '            route = C.T.route_teacher(st, ax)')
    replace('            candidate=(st.held_item[:,0]==1)',
            '            goal_evidence=j.stack([legacy_route.any((1,2)), '
            'legacy_route.any((1,2))&~route.any((1,2)), '
            'legacy_route.any((1,2))&~route.any((1,2))&~item.any((1,2))],-1)\n'
            '            candidate=(st.held_item[:,0]==1)')
    replace("ax['provenance_source'], exposure, qualification)",
            "ax['provenance_source'], exposure, qualification, goal_evidence)")
    replace('origins,exposure,qualification = d', 'origins,exposure,qualification,goal_evidence = d')
    replace('(parts,flags,od,reward,v,advantage,origins,exposure,qualification)',
            '(parts,flags,od,reward,v,advantage,origins,exposure,qualification,goal_evidence)')
    namespace = dict(vars(template), legacy_teacher=legacy_teacher)
    exec(compile(source, '<goal-kind-qualification-collector>', 'exec'), namespace)
    collect = namespace['collector'](steps, qualified=False)

    def run(*args):
        end, batch, diagnostic = collect(*args)
        counters[stream] = summary(diagnostic[1], diagnostic[6], diagnostic[-1])
        # Restore the exact existing public tuple schema; no new policy input.
        return end, batch, diagnostic[:-1]

    return run
