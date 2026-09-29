"""最小 Python Bot 插件：应用启动时显式调用 register()。"""

from __future__ import annotations

import hashlib

from qqt_rl.bots import BotAction, BotSpec


class IdleBot:
    def reset(self, context):
        self.episode_id = context.episode_id

    def act(self, observation, player_id, rng):
        return BotAction(move=4, ability=0).validate(
            observation.legal_moves, observation.legal_abilities)

    def close(self):
        return None


def register(registry):
    implementation_hash = hashlib.sha256(__file__.encode()).hexdigest()
    registry.register(BotSpec(
        id="example.idle", version="1.0.0", display_name="示例原地等待 Bot",
        runtime=("python",), capabilities={
            "deterministic": True, "batched": "none", "jittable": False,
            "async": False, "transition_observer": False, "frozen": True},
        config_schema={"type": "object", "properties": {},
                       "additionalProperties": False}, defaults={},
        identity_hash=hashlib.sha256(b"example.idle@1.0.0").hexdigest(),
        implementation_hash=implementation_hash,
        fixture_hash="0" * 64,
        provenance_hash=hashlib.sha256(
            f"example.idle@1.0.0:{implementation_hash}".encode()).hexdigest(),
    ), lambda config: IdleBot())
