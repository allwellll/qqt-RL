"""Host-only tactical labels; never part of learner observations."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from . import bun_rule_bot as rule

@dataclass(frozen=True)
class TacticalStateLabel:
    safe_attack_available: bool
    safe_escape_exists: bool
    enemy_escape_count: int
    forced_kill_state: bool
    minimum_escape_time: int


def label_mapping(mapping: Mapping[str, Any], player_id: int = 0,
                  horizon: int = 40) -> TacticalStateLabel:
    bot=rule.BunRuleBot(); bot.horizon_steps=max(16,min(40,int(horizon)))
    state=rule._parse_state(mapping); neighbors=bot._get_neighbors(state)
    danger,predicted,bomb_until=bot._predict(state)
    own=bot._survival_plan(state,player_id,danger,bomb_until,neighbors)
    enemy=bot._survival_plan(state,1-player_id,danger,bomb_until,neighbors)
    attack=bot._safe_attack(state,player_id,neighbors)
    safe_attack=bool(attack is not None and attack[0].survived and attack[2])
    minimum=max(0, horizon-own.survival_steps) if own.survived else horizon+1
    return TacticalStateLabel(safe_attack,bool(own.survived),len(enemy.safe_actions),bool(own.survived and not enemy.survived),int(minimum))


def label_batch(mappings: Sequence[Mapping[str, Any]], player_ids=None, horizon=40):
    if player_ids is None: player_ids=np.zeros(len(mappings),np.int32)
    return [label_mapping(state,int(pid),horizon) for state,pid in zip(mappings,player_ids)]


def derive_action_labels(*, legal, survivable, abilities, self_alive,
                         enemy_alive, nontrade_kill, trade, own_bomb_first,
                         enemy_escape_count, minimum_escape_time,
                         causal_kill_onset, initial_forced=False):
    legal=np.asarray(legal,np.bool_);survivable=np.asarray(survivable,np.bool_)
    abilities=np.asarray(abilities);self_alive=np.asarray(self_alive,np.bool_)
    enemy_alive=np.asarray(enemy_alive,np.bool_);nontrade_kill=np.asarray(nontrade_kill,np.bool_)
    trade=np.asarray(trade,np.bool_);own_bomb_first=np.asarray(own_bomb_first,np.bool_)
    forced=self_alive & ~enemy_alive & nontrade_kill & ~trade
    bomb_forced=(abilities==rule.ABILITY_BOMB)&forced&(not initial_forced)
    return {
        'safe_attack_available':np.bool_(np.any(legal&bomb_forced&survivable)),
        'self_survive_4s':self_alive,
        'enemy_survive_4s':enemy_alive,
        'safe_escape_exists':survivable,
        'enemy_escape_count':np.asarray(enemy_escape_count,np.int16),
        'forced_kill_state':forced,
        'bomb_creates_forced_kill':bomb_forced,
        'bomb_creates_trade':(abilities==rule.ABILITY_BOMB)&trade,
        'own_bomb_death_risk':own_bomb_first,
        'causal_kill_onset':np.asarray(causal_kill_onset,np.int16),
        'forced_kill_onset':np.where(bomb_forced,1,-1).astype(np.int16),
        'minimum_escape_time':np.asarray(minimum_escape_time,np.float32),
    }
