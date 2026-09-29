"""显式机器人注册表；不接受任意模块路径动态加载。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .schema import BotSpec


Factory = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class _Entry:
    spec: BotSpec
    factory: Factory | None


class BotRegistry:
    def __init__(self, runtime: str):
        self.runtime = runtime
        self._entries: dict[str, _Entry] = {}

    def register(self, spec: BotSpec, factory: Factory | None) -> None:
        if spec.id in self._entries:
            raise ValueError(f"duplicate bot id: {spec.id}")
        self._entries[spec.id] = _Entry(spec, factory)

    def list(self) -> list[dict[str, Any]]:
        return [entry.spec.to_dict() for _, entry in sorted(self._entries.items())]

    def _validated_config(self, spec: BotSpec, config: Mapping[str, Any] | None) -> dict[str, Any]:
        supplied = dict(config or {})
        properties = dict(spec.config_schema.get("properties", {}))
        unknown = sorted(set(supplied) - set(properties))
        if unknown and spec.config_schema.get("additionalProperties") is False:
            raise ValueError(f"unknown config keys for {spec.id}: {unknown}")
        merged = dict(spec.defaults)
        merged.update(supplied)
        for key, rule in properties.items():
            if key not in merged:
                continue
            expected = rule.get("type")
            value = merged[key]
            valid = (expected == "number" and isinstance(value, (int, float))
                     or expected == "integer" and isinstance(value, int)
                     or expected == "boolean" and isinstance(value, bool)
                     or expected == "string" and isinstance(value, str))
            if expected and not valid:
                raise ValueError(f"invalid config type for {spec.id}.{key}")
            if "enum" in rule and value not in rule["enum"]:
                raise ValueError(f"invalid config value for {spec.id}.{key}")
        return merged

    def describe(self, bot_id: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
        if bot_id not in self._entries:
            raise KeyError(f"unknown bot id: {bot_id}")
        entry = self._entries[bot_id]
        return {
            "spec": entry.spec.to_dict(),
            "config": self._validated_config(entry.spec, config),
            "implementation_hash": entry.spec.implementation_hash,
            "fixture_hash": entry.spec.fixture_hash,
        }

    def validate(self, bot_id: str, config: Mapping[str, Any] | None,
                 required_version: str | None = None,
                 required_capabilities: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if bot_id not in self._entries:
            raise KeyError(f"unknown bot id: {bot_id}")
        spec = self._entries[bot_id].spec
        if required_version is not None and required_version != spec.version:
            raise ValueError(f"incompatible bot version: {required_version}")
        for name, expected in (required_capabilities or {}).items():
            if spec.capabilities.get(name) != expected:
                raise ValueError(f"capability mismatch: {name}")
        return self._validated_config(spec, config)

    def create(self, bot_id: str, config: Mapping[str, Any] | None,
               required_version: str | None = None,
               required_capabilities: Mapping[str, bool] | None = None):
        if bot_id not in self._entries:
            raise KeyError(f"unknown bot id: {bot_id}")
        entry = self._entries[bot_id]
        if self.runtime not in entry.spec.runtime or entry.factory is None:
            raise ValueError(f"bot {bot_id} does not support runtime {self.runtime}")
        validated = self.validate(
            bot_id, config, required_version, required_capabilities)
        return entry.factory(validated)
