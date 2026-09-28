"""Host-side adapters for frozen Bun code opponents.

Code opponents are intentionally kept outside learner gradients and JIT.  The
adapter batches Python decisions on host and exposes an explicit provenance
record so training manifests can distinguish code-policy strata from learned
checkpoint opponents.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from .bun_tactical_family import TacticalFamilyBot, TacticalFamilyConfig

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
    source = Path(os.environ.get(
        "BUN_TACTICAL_BOT_PATH", DEFAULT_FROZEN_TACTICAL_BOT)).resolve()
    if not source.is_file():
        raise RuntimeError(f"frozen tactical bot source missing: {source}")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    module_name = f"jax_bomb._bun_rule_bot_frozen_{source_hash}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen tactical bot source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, source


bun_rule_bot, _TACTICAL_BOT_SOURCE = _load_tactical_bot()


def _module_sha256() -> str:
    digest = hashlib.sha256()
    with _TACTICAL_BOT_SOURCE.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tactical_bot_provenance() -> dict[str, Any]:
    sha256 = _module_sha256()
    family_config = TacticalFamilyConfig.from_mapping(json.loads(
        os.environ.get("BUN_TACTICAL_FAMILY_JSON", "{}")))
    family_path = Path(__file__).with_name("bun_tactical_family.py")
    family_sha256 = hashlib.sha256(family_path.read_bytes()).hexdigest()
    return {
        "name": "bun_rule_tactical_40tick",
        "module": "jax_bomb.bun_rule_bot",
        "source_path": str(_TACTICAL_BOT_SOURCE),
        "module_sha256": sha256,
        "expected_sha256": EXPECTED_TACTICAL_BOT_SHA256,
        "hash_verified": sha256 == EXPECTED_TACTICAL_BOT_SHA256,
        "family_config": asdict(family_config),
        "family_module_sha256": family_sha256,
        "opponent_identity_hash": family_config.identity(sha256, family_sha256),
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
        self._family_config = TacticalFamilyConfig.from_mapping(json.loads(
            os.environ.get("BUN_TACTICAL_FAMILY_JSON", "{}")))
        self._family_bot = TacticalFamilyBot(self._family_config)

    @property
    def provenance(self) -> dict[str, Any]:
        return tactical_bot_provenance()

    def decide_batch(self, states: Any) -> np.ndarray:
        mappings = [bun_rule_bot.state_from_bun_state(state)
                    for state in _unbatch(states)]
        player_ids = np.full((len(mappings),), self.player_id, np.int32)
        if self._family_config.name == "full_v2":
            return bun_rule_bot.decide_batch(mappings, player_ids)
        return np.stack([
            self._family_bot.decide(mapping, int(player_id))
            for mapping, player_id in zip(mappings, player_ids)
        ]).astype(np.int32, copy=False)


__all__ = [
    "DEFAULT_EXPECTED_TACTICAL_BOT_SHA256",
    "EXPECTED_TACTICAL_BOT_SHA256",
    "FrozenTacticalOpponent",
    "clear_destructible_bricks",
    "tactical_bot_provenance",
]
