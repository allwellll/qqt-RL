"""跨 Python/浏览器共享的机器人可序列化契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class BotSpec:
    id: str
    version: str
    display_name: str
    runtime: tuple[str, ...]
    capabilities: Mapping[str, Any]
    config_schema: Mapping[str, Any]
    defaults: Mapping[str, Any]
    identity_hash: str
    implementation_hash: str
    fixture_hash: str
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runtime"] = list(self.runtime)
        value["capabilities"] = dict(self.capabilities)
        value["config_schema"] = dict(self.config_schema)
        value["defaults"] = dict(self.defaults)
        return value


@dataclass(frozen=True)
class BotContext:
    episode_id: str
    seed: int
    ruleset: str = "bun"
    max_ticks: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "qqt.bot.context/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BotObservation:
    tick: int
    state: Mapping[str, Any]
    legal_moves: Sequence[int] = (0, 1, 2, 3, 4)
    legal_abilities: Sequence[int] = (0, 1, 2)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "qqt.bot.observation/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BotAction:
    move: int
    ability: int

    def validate(self, legal_moves: Sequence[int] = (0, 1, 2, 3, 4),
                 legal_abilities: Sequence[int] = (0, 1, 2)) -> "BotAction":
        if self.move not in range(5) or self.ability not in range(3):
            raise ValueError(f"invalid Bun action: {(self.move, self.ability)}")
        if self.move not in legal_moves or self.ability not in legal_abilities:
            raise ValueError(f"illegal Bun action: {(self.move, self.ability)}")
        return self

    def to_dict(self) -> dict[str, int]:
        return {"move": self.move, "ability": self.ability}


@runtime_checkable
class Bot(Protocol):
    def reset(self, context: BotContext) -> None: ...
    def act(self, observation: BotObservation, player_id: int, rng: Any) -> BotAction: ...
    def close(self) -> None: ...
