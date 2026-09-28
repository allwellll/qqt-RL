"""Deterministic, collision-free seed namespaces for Bun critic relabel data."""

from __future__ import annotations


SPLITS = ("train", "validation", "test")
V7_BASE = 3_000_000_000
V7_CANDIDATE_STRIDE = 100_000_000
V7_CYCLE_STRIDE = 100_000
V7_SPLIT_STRIDE = 10_000


def _plan(base: int, split_stride: int, train_states: int) -> dict[str, list[int]]:
    counts = (train_states, 1, 1)
    return {
        split: [base + split_index * split_stride + item for item in range(count)]
        for split_index, (split, count) in enumerate(zip(SPLITS, counts))
    }


def legacy_critic_seed_plan(
        learner_seed: int, cycle: int, *, train_states: int) -> dict[str, list[int]]:
    return _plan(learner_seed + cycle * 10_000, 100_000, train_states)


def critic_seed_base_v7(candidate_slot: int, learner_seed: int, cycle: int) -> int:
    # raw seed 与 JAX 实际使用的 uint32(seed) 都必须跨 candidate/cycle/split 唯一。
    # 因此 namespace 从低于 2**32 的固定区域分段，而不是继续叠加旧实验的大 seed。
    if candidate_slot not in range(4):
        raise ValueError("candidate_slot must be in [0, 3]")
    if cycle not in range(1, 361):
        raise ValueError("cycle must be in [1, 360]")
    if learner_seed <= 0:
        raise ValueError("learner_seed must be positive")
    return V7_BASE + candidate_slot * V7_CANDIDATE_STRIDE + cycle * V7_CYCLE_STRIDE


def critic_seed_plan_v7(
        candidate_slot: int, learner_seed: int, cycle: int, *,
        train_states: int) -> dict[str, list[int]]:
    # split 使用独立 10k 子空间，concat 时仍会再次执行严格泄漏检查，双层防护。
    return _plan(
        critic_seed_base_v7(candidate_slot, learner_seed, cycle),
        V7_SPLIT_STRIDE,
        train_states,
    )
