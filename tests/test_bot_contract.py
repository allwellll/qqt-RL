import json
from pathlib import Path

import numpy as np

from qqt_rl.bots import (
    BotAction,
    BotContext,
    BotObservation,
    BotRegistry,
    BotSpec,
    create_default_registry,
)
from jax_bomb.bun_frozen_opponents import tactical_bot_provenance

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/bot_contract_cases.json").read_text())


def test_action_schema_and_legality_validation():
    action = BotAction(move=4, ability=0)
    assert action.to_dict() == {"move": 4, "ability": 0}
    action.validate(legal_moves=(0, 4), legal_abilities=(0, 1))
    for invalid in (BotAction(-1, 0), BotAction(0, 3)):
        try:
            invalid.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action accepted")


def test_registry_rejects_duplicate_unknown_config_version_and_capability():
    registry = BotRegistry(runtime="python")
    spec = BotSpec(
        id="example.bot", version="1.0.0", display_name="Example",
        runtime=("python",), capabilities={"deterministic": True},
        config_schema={"type": "object", "properties": {}, "additionalProperties": False},
        defaults={}, identity_hash="a" * 64, implementation_hash="b" * 64,
        fixture_hash="c" * 64, provenance_hash="d" * 64)
    registry.register(spec, lambda config: object())
    assert registry.validate("example.bot", {}) == {}
    for operation in (
        lambda: registry.register(spec, lambda config: object()),
        lambda: registry.create("missing", {}),
        lambda: registry.create("example.bot", {"unknown": 1}),
        lambda: registry.create("example.bot", {}, required_version="2.0.0"),
        lambda: registry.create("example.bot", {}, required_capabilities={"async": True}),
        lambda: registry.validate("example.bot", {"unknown": 1}),
    ):
        try:
            operation()
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError("invalid registry operation accepted")


def test_builtin_tactical_bot_matches_cross_language_fixture_and_manifest():
    registry = create_default_registry(runtime="python")
    record = registry.describe("bun.tactical_v2", {})
    assert record["spec"]["identity_hash"] == FIXTURE["tactical_identity_hash"]
    assert record["spec"]["capabilities"]["frozen"] is True
    assert record["spec"]["capabilities"]["batched"] == "native_host"
    assert len(record["implementation_hash"]) == 64
    assert record["fixture_hash"] == FIXTURE["fixture_hash"]
    bot = registry.create("bun.tactical_v2", {})
    bot.reset(BotContext(episode_id="fixture", seed=7))
    case = FIXTURE["action_cases"][0]
    action = bot.act(BotObservation(tick=0, state=case["state"]), case["player_id"], 7)
    action.validate()
    assert action.to_dict() == case["expected_action"]


def test_random_roam_and_browser_model_share_contract():
    registry = create_default_registry(runtime="python")
    roam = registry.create("bun.random_roam", {"allow_idle": False})
    observation = BotObservation(
        tick=0, state={}, legal_moves=(0, 1, 4), legal_abilities=(0,))
    first = roam.act(observation, 1, 123)
    second = roam.act(observation, 1, 123)
    assert first == second
    model = next(row for row in registry.list() if row["id"] == "bun.browser_model")
    assert model["capabilities"]["async"] is True
    try:
        registry.create("bun.browser_model", {})
    except ValueError as error:
        assert "runtime" in str(error)
    else:
        raise AssertionError("browser-only model created in Python runtime")


def test_frozen_opponent_provenance_embeds_registry_contract():
    provenance = tactical_bot_provenance()
    assert provenance["bot_id"] == "bun.tactical_v2"
    assert provenance["bot_spec"]["identity_hash"] == FIXTURE["tactical_identity_hash"]
    assert provenance["bot_config"]["name"] == "full_v2"
    assert len(provenance["implementation_hash"]) == 64
    assert len(provenance["fixture_hash"]) == 64
