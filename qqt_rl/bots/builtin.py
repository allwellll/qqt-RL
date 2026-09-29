"""内置 Bun 机器人适配器及稳定身份。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jax_bomb import bun_rule_bot
from jax_bomb.bun_tactical_family import TacticalFamilyBot, TacticalFamilyConfig

from .registry import BotRegistry
from .schema import BotAction, BotContext, BotObservation, BotSpec

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PAYLOAD = {
    "contract": "qqt.bot/v1", "id": "bun.tactical_v2",
    "semantic_fixture": "bun_rule_bot_v2", "version": "2.0.0",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode())


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fixture_hash(path: Path) -> str:
    payload = json.loads(path.read_text())
    return _canonical_hash({
        "schema": payload["schema"], "action_cases": payload["action_cases"]})


class TacticalV2Adapter:
    def __init__(self, config: dict[str, Any]):
        self.config = TacticalFamilyConfig.from_mapping(config)
        self.bot = TacticalFamilyBot(self.config)

    def reset(self, context: BotContext) -> None:
        self.bot.reset()

    def act(self, observation: BotObservation, player_id: int, rng: Any) -> BotAction:
        action = self.bot.decide(observation.state, player_id)
        return BotAction(int(action[0]), int(action[1])).validate(
            observation.legal_moves, observation.legal_abilities)

    def act_batch(self, observations, player_ids, rngs=None) -> np.ndarray:
        # batch 是 host emulation：每个环境独立实例，避免 FSM phase/route cache 串扰。
        return np.asarray([
            TacticalFamilyBot(self.config).decide(observation.state, int(player_id))
            for observation, player_id in zip(observations, player_ids)
        ], np.int32)

    def observe_transition(self, event: dict[str, Any],
                           next_observation: BotObservation, player_id: int) -> None:
        self.bot.bot.observe_transition(event, next_observation.state, player_id)

    def close(self) -> None:
        return None


class RandomRoamBot:
    def __init__(self, config: dict[str, Any]):
        self.allow_idle = bool(config["allow_idle"])

    def reset(self, context: BotContext) -> None:
        return None

    def act(self, observation: BotObservation, player_id: int, rng: Any) -> BotAction:
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(int(rng))
        moves = [move for move in observation.legal_moves
                 if self.allow_idle or move != 4]
        if not moves:
            moves = list(observation.legal_moves)
        return BotAction(int(generator.choice(moves)), 0).validate(
            observation.legal_moves, observation.legal_abilities)

    def close(self) -> None:
        return None


def create_default_registry(runtime: str) -> BotRegistry:
    fixture = ROOT / "tests/fixtures/bot_contract_cases.json"
    tactical_source = ROOT / "jax_bomb/bun_rule_bot.py"
    family_source = ROOT / "jax_bomb/bun_tactical_family.py"
    fixture_hash = _fixture_hash(fixture)
    implementation_hash = _canonical_hash({
        "rule": _file_hash(tactical_source), "family": _file_hash(family_source)})
    identity_hash = _canonical_hash(IDENTITY_PAYLOAD)
    capabilities = {
        "deterministic": True, "batched": "native_host", "jittable": False,
        "async": False, "transition_observer": True, "frozen": True,
    }
    tactical_defaults = TacticalFamilyConfig().__dict__
    tactical_schema = {"type": "object", "additionalProperties": False,
                       "properties": {
                           "name": {"type": "string"},
                           "horizon": {"type": "integer"},
                           "aggression": {"type": "number"},
                           "escape_margin": {"type": "integer"},
                           "tie_break": {"type": "integer"},
                           "bomb_threshold": {"type": "integer"},
                           "positioning_preference": {"type": "string"},
                           "structural_mode": {"type": "string"},
                       }}
    tactical_spec = BotSpec(
        id="bun.tactical_v2", version="2.0.0",
        display_name="Bun Tactical v2", runtime=("python", "browser"),
        capabilities=capabilities, config_schema=tactical_schema,
        defaults=tactical_defaults, identity_hash=identity_hash,
        implementation_hash=implementation_hash, fixture_hash=fixture_hash,
        provenance_hash=_canonical_hash({
            "identity": identity_hash, "implementation": implementation_hash,
            "fixture": fixture_hash}),
    )
    roam_identity = _canonical_hash({
        "contract": "qqt.bot/v1", "id": "bun.random_roam", "version": "1.0.0"})
    roam_impl = _file_hash(Path(__file__))
    roam_spec = BotSpec(
        id="bun.random_roam", version="1.0.0", display_name="随机漫游 Bot",
        runtime=("python", "browser"), capabilities={
            "deterministic": False, "batched": "none", "jittable": False,
            "async": False, "transition_observer": False, "frozen": True},
        config_schema={"type": "object", "additionalProperties": False,
                       "properties": {"allow_idle": {"type": "boolean"}}},
        defaults={"allow_idle": True}, identity_hash=roam_identity,
        implementation_hash=roam_impl, fixture_hash=fixture_hash,
        provenance_hash=_canonical_hash({"identity": roam_identity, "implementation": roam_impl}),
    )
    model_identity = _canonical_hash({
        "contract": "qqt.bot/v1", "id": "bun.browser_model", "version": "1.0.0"})
    model_spec = BotSpec(
        id="bun.browser_model", version="1.0.0", display_name="浏览器模型 Bot",
        runtime=("browser",), capabilities={
            "deterministic": True, "batched": "none", "jittable": False,
            "async": True, "transition_observer": False, "frozen": True},
        config_schema={"type": "object", "additionalProperties": False,
                       "properties": {}}, defaults={}, identity_hash=model_identity,
        implementation_hash="0" * 64, fixture_hash=fixture_hash,
        provenance_hash=_canonical_hash({"identity": model_identity, "runtime": "browser"}),
    )
    registry = BotRegistry(runtime)
    registry.register(tactical_spec, lambda config: TacticalV2Adapter(config))
    registry.register(roam_spec, lambda config: RandomRoamBot(config))
    registry.register(model_spec, None)
    return registry
