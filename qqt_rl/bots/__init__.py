"""稳定的 Bun Bot 生命周期、schema 与显式注册入口。"""

from .builtin import create_default_registry
from .registry import BotRegistry
from .schema import Bot, BotAction, BotContext, BotObservation, BotSpec

__all__ = [
    "Bot", "BotAction", "BotContext", "BotObservation", "BotRegistry", "BotSpec",
    "create_default_registry",
]
