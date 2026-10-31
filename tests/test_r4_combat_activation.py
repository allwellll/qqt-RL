import inspect


def test_worker_bounded_no_reward_edit():
    from scripts.r4_combat_train import worker_source
    s = worker_source()
    compile(s, 'worker', 'exec')
    assert "runs/r4_combat_20261031" in s
    assert "protocol['round14_collector_seed'],.10,0.,spawns" not in s
    assert "spec['curriculum_fraction']" in s
    assert 'r4_lesson_ticks' in s
    assert "min(row['update_l2'].values())>0" in s


def test_live_fresh_and_reset_share_lessons():
    from scripts import r3_items_collector as module
    from scripts.r4_combat_collector import install
    original_reward = module.transition
    install(module)
    assert module.transition is original_reward
    assert module.fresh is module._impl.fresh
    assert module._base._impl is module._impl
    # The compiled function references immutable module literals.
    s = inspect.getsource(__import__('scripts.r4_combat_collector', fromlist=['install']).install)
    assert '.25/.7' in s
    assert 'LESSON_COMBAT_KILL' in s and 'LESSON_KILL_RUSH' in s
    assert 'bun_carried=' not in s
