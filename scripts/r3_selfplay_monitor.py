"""Passive on-policy natural self-play counters; no reward/observation changes."""

def state(n):
    import jax.numpy as j
    return dict(entered=j.zeros((n, 2), bool), bombs=j.zeros((n, 2), j.int32),
                interception_pending=j.zeros((n, 2), bool), loops=j.zeros((n, 2), j.int32))


def advance(before, after, info, aux):
    import jax
    import jax.numpy as j
    from scripts.carrier_mechanism_engine import env
    m = aux['combat_mix']['sp_monitor']
    n = len(before.t); rows = j.arange(n)[:, None]; players = j.arange(2)[None]
    cells = j.clip(j.floor(after.pos).astype(j.int32), j.array([0, 0]), j.array([12, 14]))
    cages = j.stack([env._base_mask(1), env._base_mask(0)])
    active = ~aux['terminal']
    entered = m['entered'] | (active[:, None] & after.alive & cages[players, cells[..., 0], cells[..., 1]])
    bombs = m['bombs'] + (active[:, None] & info['bomb_placed']).astype(j.int32)
    intercepted = info['surviving_kill'] & (before.bun_carried[:, ::-1] == players)
    pending = m['interception_pending'] | (active[:, None] & intercepted)
    # Several identical-origin packs can already remain at home. A kill is
    # not a completed recovery until native home inventory actually grows.
    restored = after.bun_stored[rows, players, players] > before.bun_stored[rows, players, players]
    loops = m['loops'] + (active[:, None] & pending & restored).astype(j.int32)
    pending &= ~restored
    first = active & info['h_route_terminal']
    # Never mix genuine mid-game Bank exposure or clear-arena resets into
    # natural full-episode denominators.
    natural = (aux['provenance_source'] < 0) & ~aux['combat_mix']['scene']
    kinds = aux['combat_mix']['kind']; seats = aux['combat_mix']['learner_seat']
    weights = jax.nn.one_hot(kinds-3, 3)[:, :, None] * jax.nn.one_hot(seats, 2)[:, None, :]
    weights *= (first & natural & (kinds >= 3))[:, None, None]
    # Collector diagnostics reduce the environment axis exactly once. Keep
    # that axis here, as every inherited reward/diagnostic metric does.
    count = lambda value: weights*value[:, None, None]
    winner = info['winner']
    metrics = dict(r3_sp_episodes=count(j.ones(n)), r3_sp_draws=count(winner == -1),
        r3_sp_unresolved=count(winner < -1),
        r3_sp_wins=count(winner == seats), r3_sp_losses=count(winner == 1-seats),
        r3_sp_passive=count(~entered.any(-1)),
        r3_sp_own_zero_bomb=count(bombs[j.arange(n), seats] == 0),
        r3_sp_both_zero_bomb=count((bombs == 0).all(-1)),
        r3_sp_custody_loops=count(loops.sum(-1)))
    # Native delivery events, not victory-by-HP or inferred intention.
    pool = jax.nn.one_hot(kinds-3, 3)[:, :, None] * jax.nn.one_hot(seats, 2)[:, None, :]
    delivery = info['capture'][j.arange(n), seats] & active & natural & (kinds >= 3)
    metrics['r3_sp_active_deliveries'] = pool*delivery[:, None, None]
    fixed = jax.nn.one_hot(j.arange(n) % 3, 3)[:, :, None] * jax.nn.one_hot(seats, 2)[:, None, :]
    fixed *= (first & natural & (kinds == 5))[:, None, None]
    for name, value in [('games', j.ones(n)), ('wins', winner == seats),
                        ('draws', winner == -1), ('unresolved', winner < -1),
                        ('losses', winner == 1-seats)]:
        metrics['r3_sp_fixed_'+name] = fixed*value[:, None, None]
    return dict(entered=entered, bombs=bombs, interception_pending=pending, loops=loops), metrics


def install(module):
    if getattr(module, '_sp_monitor_installed', False):
        return
    # A static opponent identity per environment lasts for the complete game;
    # rotating one historical actor every rollout would confound the curve.
    from types import ModuleType
    source = module.src
    anchor = "for name,actor in [('current',p['actor']),('ring',ring),('fixed',fixed)]:"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "for name,actor in [('current',p['actor']),('ring',ring),('fixed0',fixed[0]),('fixed1',fixed[1]),('fixed2',fixed[2])]:")
    anchor = ";groups['bot']=np.flatnonzero(kinds<3)"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor+"\n    for identity in range(3):groups['fixed'+str(identity)]=np.flatnonzero((kinds==5)&(np.arange(count)%3==identity))")
    impl = ModuleType('r3_sp_private_collector')
    exec(compile(source, __file__, 'exec'), impl.__dict__)
    for name in ('fresh', 'transition', 'metadata', 'inputs', 'prediction'):
        setattr(impl, name, getattr(module._impl, name))
    module._impl = module._base._impl = impl
    for name in ('choose', 'choose_with_diagnostics'):
        setattr(module, name, getattr(impl, name))
    metadata, transition = module.metadata, module.transition
    def monitored_metadata(st, *args, **kwargs):
        return dict(metadata(st, *args, **kwargs), sp_monitor=state(len(st.t)))
    def monitored_transition(st, ax, *args, **kwargs):
        after, na, reward, done, info, parts, metrics = transition(st, ax, *args, **kwargs)
        monitor, counts = advance(st, after, info, ax)
        na = dict(na, combat_mix=dict(na['combat_mix'], sp_monitor=monitor))
        return after, na, reward, done, info, parts, dict(metrics, **counts)
    module.metadata = module._impl.metadata = monitored_metadata
    module.transition = module._impl.transition = monitored_transition
    module._sp_monitor_installed = True
