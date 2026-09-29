"""Host-side adapters for frozen Bun code opponents.

Code opponents are intentionally kept outside learner gradients and JIT.  The
adapter batches Python decisions on host and exposes an explicit provenance
record so training manifests can distinguish code-policy strata from learned
checkpoint opponents.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from . import bun_rule_bot
from .bun_tactical_family import TacticalFamilyConfig

DEFAULT_EXPECTED_TACTICAL_BOT_SHA256 = (
    "021386cd927fe8c1667750b70a3c1df86d9375c3357a6d2057696fdff7365710"
)
EXPECTED_TACTICAL_BOT_SHA256 = os.environ.get(
    "BUN_TACTICAL_BOT_EXPECTED_SHA256",
    DEFAULT_EXPECTED_TACTICAL_BOT_SHA256,
)
DEFAULT_FROZEN_TACTICAL_BOT = (
    Path(__file__).with_name("bun_rule_bot.py")
)


def _load_tactical_bot():
    source = DEFAULT_FROZEN_TACTICAL_BOT.resolve()
    requested = Path(os.environ.get("BUN_TACTICAL_BOT_PATH", source)).resolve()
    if requested != source:
        raise RuntimeError(
            "arbitrary tactical bot module paths are forbidden; register a BotSpec instead")
    if not source.is_file():
        raise RuntimeError(f"frozen tactical bot source missing: {source}")
    return bun_rule_bot, source


bun_rule_bot, _TACTICAL_BOT_SOURCE = _load_tactical_bot()


def _module_sha256() -> str:
    digest = hashlib.sha256()
    with _TACTICAL_BOT_SOURCE.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tactical_bot_provenance() -> dict[str, Any]:
    from qqt_rl.bots import create_default_registry

    sha256 = _module_sha256()
    bot_id = os.environ.get("BUN_OPPONENT_BOT_ID", "bun.tactical_v2")
    config_text = os.environ.get(
        "BUN_OPPONENT_BOT_CONFIG_JSON",
        os.environ.get("BUN_TACTICAL_FAMILY_JSON", "{}"))
    config = json.loads(config_text)
    family_config = (TacticalFamilyConfig.from_mapping(config)
                     if bot_id == "bun.tactical_v2" else None)
    bot_record = create_default_registry("python").describe(
        bot_id, asdict(family_config) if family_config is not None else config)
    family_path = Path(__file__).with_name("bun_tactical_family.py")
    family_sha256 = hashlib.sha256(family_path.read_bytes()).hexdigest()
    module_sha256 = sha256 if bot_id == "bun.tactical_v2" else bot_record["implementation_hash"]
    expected_sha256 = (EXPECTED_TACTICAL_BOT_SHA256
                       if bot_id == "bun.tactical_v2" else module_sha256)
    identity_payload = {
        "identity_hash": bot_record["spec"]["identity_hash"],
        "implementation_hash": bot_record["implementation_hash"],
        "fixture_hash": bot_record["fixture_hash"],
        "config": bot_record["config"],
    }
    identity_hash = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "name": bot_record["spec"]["display_name"],
        "module": bot_record["spec"]["id"],
        "source_path": str(_TACTICAL_BOT_SOURCE),
        "module_sha256": module_sha256,
        "expected_sha256": expected_sha256,
        "hash_verified": module_sha256 == expected_sha256,
        "family_config": (asdict(family_config)
                          if family_config is not None else config),
        "family_module_sha256": family_sha256,
        "opponent_identity_hash": identity_hash,
        "bot_id": bot_id,
        "bot_spec": bot_record["spec"],
        "bot_config": bot_record["config"],
        "implementation_hash": bot_record["implementation_hash"],
        "fixture_hash": bot_record["fixture_hash"],
        "horizon_steps": bun_rule_bot.HORIZON_STEPS,
        "tick_hz": bun_rule_bot.TICK_HZ,
        "jittable": False,
        "execution": "host_batched_rollout",
        "learner_gradient": False,
        "destructible_bricks_cleared": True,
    }


def clear_destructible_bricks(states: Any) -> Any:
    """Clear destructible bricks while preserving native permanent walls."""
    core = states.core._replace(
        brick=jnp.zeros_like(states.core.brick),
        brick_linger=jnp.zeros_like(states.core.brick_linger),
    )
    return states._replace(core=core)


def _unbatch(states: Any) -> list[Any]:
    host = jax.device_get(states)
    count = int(np.asarray(host.core.pos).shape[0])
    return [jax.tree.map(lambda value, index=index: value[index], host)
            for index in range(count)]


@dataclass
class FrozenTacticalOpponent:
    """Stable batch interface around the 40-tick Python tactical policy."""

    player_id: int = 1
    verify_hash: bool = True

    def __post_init__(self) -> None:
        if self.player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        provenance = tactical_bot_provenance()
        if self.verify_hash and not provenance["hash_verified"]:
            raise RuntimeError(
                "frozen tactical bot hash changed: "
                f"{provenance['module_sha256']}")
        bot_id = os.environ.get("BUN_OPPONENT_BOT_ID", "bun.tactical_v2")
        config_text = os.environ.get(
            "BUN_OPPONENT_BOT_CONFIG_JSON",
            os.environ.get("BUN_TACTICAL_FAMILY_JSON", "{}"))
        config = json.loads(config_text)
        self._family_config = (TacticalFamilyConfig.from_mapping(config)
                               if bot_id == "bun.tactical_v2" else None)
        from qqt_rl.bots import BotContext, create_default_registry
        self._bot = create_default_registry("python").create(
            bot_id, asdict(self._family_config) if self._family_config is not None else config,
            required_capabilities={"frozen": True})
        self._bot.reset(BotContext(
            episode_id="frozen-opponent", seed=0,
            metadata={"learner_gradient": False}))

    @property
    def provenance(self) -> dict[str, Any]:
        return tactical_bot_provenance()

    def decide_batch(self, states: Any) -> np.ndarray:
        mappings = [bun_rule_bot.state_from_bun_state(state)
                    for state in _unbatch(states)]
        player_ids = np.full((len(mappings),), self.player_id, np.int32)
        from qqt_rl.bots import BotObservation
        observations = [BotObservation(tick=0, state=mapping) for mapping in mappings]
        if hasattr(self._bot, "act_batch"):
            return self._bot.act_batch(observations, player_ids)
        actions = [
            self._bot.act(observation, int(player_id), index)
            for index, (observation, player_id) in enumerate(
                zip(observations, player_ids))
        ]
        return np.asarray([[action.move, action.ability] for action in actions], np.int32)


__all__ = [
    "DEFAULT_EXPECTED_TACTICAL_BOT_SHA256",
    "EXPECTED_TACTICAL_BOT_SHA256",
    "FrozenTacticalOpponent",
    "clear_destructible_bricks",
    "tactical_bot_provenance",
]
