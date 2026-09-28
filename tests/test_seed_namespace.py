from jax_bomb.bun_seed_namespace import SPLITS, critic_seed_plan_v7


def test_seed_namespace_is_collision_free_for_all_candidates_and_cycles():
    raw = {}
    effective = {}
    for slot in range(4):
        learner_seed = 202609280001 + slot * 100
        for cycle in range(1, 361):
            plan = critic_seed_plan_v7(slot, learner_seed, cycle, train_states=2)
            assert tuple(plan) == SPLITS
            for split, seeds in plan.items():
                for seed in seeds:
                    assert seed not in raw, (seed, raw[seed], (slot, cycle, split))
                    effective_seed = seed & 0xFFFFFFFF
                    assert effective_seed not in effective
                    raw[seed] = (slot, cycle, split)
                    effective[effective_seed] = (slot, cycle, split)


def test_seed_namespace_rejects_cycle_outside_formal_range():
    for cycle in (0, 361):
        try:
            critic_seed_plan_v7(0, 1, cycle, train_states=2)
        except ValueError as error:
            assert "[1, 360]" in str(error)
        else:
            raise AssertionError("out-of-range cycle accepted")
