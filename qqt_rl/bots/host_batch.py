"""JAX-free host batch work for the tactical-opponent eval: rule-bot actions + funnel labels.

Both computations are stateless per call (a fresh bot per game), so splitting a
tick's games across processes is exactly equivalent to the serial loop.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from jax_bomb.bun_tactical_family import TacticalFamilyConfig
from jax_bomb.bun_tactical_labels import label_batch

_BOT = None


def opponent_spec_from_env() -> tuple[str, dict[str, Any]]:
    """Same bot id/config resolution as FrozenTacticalOpponent."""
    bot_id = os.environ.get("BUN_OPPONENT_BOT_ID", "bun.tactical_v2")
    config = json.loads(os.environ.get(
        "BUN_OPPONENT_BOT_CONFIG_JSON", os.environ.get("BUN_TACTICAL_FAMILY_JSON", "{}")))
    if bot_id == "bun.tactical_v2":
        config = asdict(TacticalFamilyConfig.from_mapping(config))
    return bot_id, config


def init_worker(bot_id: str, config: dict[str, Any]) -> None:
    global _BOT
    from qqt_rl.bots import BotContext, create_default_registry
    _BOT = create_default_registry("python").create(
        bot_id, config, required_capabilities={"frozen": True})
    _BOT.reset(BotContext(episode_id="frozen-opponent", seed=0,
                          metadata={"learner_gradient": False}))


def bot_actions(mappings: Sequence[dict], player_id: int) -> np.ndarray:
    from qqt_rl.bots import BotObservation
    observations = [BotObservation(tick=0, state=mapping) for mapping in mappings]
    player_ids = np.full((len(mappings),), player_id, np.int32)
    if hasattr(_BOT, "act_batch"):
        return np.asarray(_BOT.act_batch(observations, player_ids), np.int32).reshape(-1, 2)
    actions = [_BOT.act(observation, player_id, index)
               for index, observation in enumerate(observations)]
    return np.asarray([[a.move, a.ability] for a in actions], np.int32).reshape(-1, 2)


def process_chunk(task: tuple[Sequence[dict], int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bot actions [n,2], labels [n,3] = safe_attack, enemy_escape_count, self_escape)."""
    mappings, bot_player, label_player = task
    actions = bot_actions(mappings, bot_player)
    labels = label_batch(mappings, np.full(len(mappings), label_player, np.int32))
    packed = np.asarray([[l.safe_attack_available, l.enemy_escape_count, l.safe_escape_exists]
                         for l in labels], np.int32).reshape(-1, 3)
    return actions, packed


class HostBatcher:
    """Runs process_chunk serially (workers=0) or over a spawn pool (no JAX in workers)."""

    def __init__(self, workers: int, bot_player: int = 1, label_player: int = 0):
        self.bot_player, self.label_player = bot_player, label_player
        self.workers = workers
        spec = opponent_spec_from_env()
        if workers > 0:
            import multiprocessing
            self._pool = multiprocessing.get_context("spawn").Pool(
                workers, initializer=init_worker, initargs=spec)
        else:
            self._pool = None
            init_worker(*spec)

    def __call__(self, mappings: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        if self._pool is None:
            return process_chunk((mappings, self.bot_player, self.label_player))
        size = max(1, -(-len(mappings) // (self.workers * 2)))
        tasks = [(mappings[i:i + size], self.bot_player, self.label_player)
                 for i in range(0, len(mappings), size)]
        parts = self._pool.map(process_chunk, tasks)
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
