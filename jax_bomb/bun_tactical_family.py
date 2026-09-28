"""Parameterized frozen tactical-v2 family and structurally distinct held-out bot."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json
from typing import Any, Mapping
import numpy as np
from . import bun_rule_bot as base

@dataclass(frozen=True)
class TacticalFamilyConfig:
    name: str = 'full_v2'
    horizon: int = 40
    aggression: float = 1.0
    escape_margin: int = 0
    tie_break: int = 0
    bomb_threshold: int = 6
    positioning_preference: str = 'chase'
    structural_mode: str = 'v2'

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        return cls(**({} if value is None else dict(value)))

    def identity(self, base_sha256: str, family_sha256: str) -> str:
        payload={'base_sha256':base_sha256,'family_sha256':family_sha256,'config':asdict(self)}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class TacticalFamilyBot:
    def __init__(self, config: TacticalFamilyConfig):
        self.config=config
        self.bot=base.BunRuleBot()
        self.bot.horizon_steps=max(16,min(40,int(config.horizon)))

    def reset(self): self.bot.reset()

    def _fallback_move(self, mapping, player_id):
        state=base._parse_state(mapping); neighbors=self.bot._get_neighbors(state)
        danger,predicted,bomb_until=self.bot._predict(state)
        plan=self.bot._survival_plan(state,player_id,danger,bomb_until,neighbors)
        actions=plan.safe_actions or (plan.first_action,)
        player=state.players[player_id]; enemy=state.players[1-player_id]
        order=[0,1,2,3,base.MOVE_IDLE]
        shift=self.config.tie_break%4; order=order[shift:4]+order[:shift]+[base.MOVE_IDLE]
        rank={a:i for i,a in enumerate(order)}; scored=[]
        start=player.row*state.width+player.col
        for action in actions:
            target=start if action==base.MOVE_IDLE else neighbors[start][action]
            if target<0: continue
            row,col=divmod(target,state.width); dist=abs(row-enemy.row)+abs(col-enemy.col)
            if self.config.positioning_preference=='flank': score=abs(row-enemy.row)-0.4*abs(col-enemy.col)
            elif self.config.positioning_preference=='space': score=dist+0.2*sum(v>=0 for v in neighbors[target])
            else: score=-dist
            scored.append((score,-rank.get(action,9),action))
        return max(scored)[2] if scored else base.MOVE_IDLE

    def decide(self, mapping, player_id=0):
        if self.config.structural_mode=='heldout_evasive_counter':
            return self._heldout(mapping,player_id)
        decision=self.bot.analyze(mapping,player_id); action=decision.action.copy()
        if self.config.name=='full_v2': return action
        state=base._parse_state(mapping); player=state.players[player_id]; enemy=state.players[1-player_id]
        distance=abs(player.row-enemy.row)+abs(player.col-enemy.col)
        signature=(player.row*97+player.col*31+enemy.row*17+enemy.col*13+len(state.bombs)*7+self.config.tie_break)%1000
        allow=(signature/1000.0)<self.config.aggression and distance<=self.config.bomb_threshold
        if int(action[1])==base.ABILITY_BOMB and not allow:
            action=np.asarray([self._fallback_move(mapping,player_id),base.ABILITY_NONE],np.int32)
        elif int(action[1])==base.ABILITY_NONE and self.config.positioning_preference!='chase' and not decision.claimed_escape:
            action=np.asarray([self._fallback_move(mapping,player_id),base.ABILITY_NONE],np.int32)
        return action

    def _heldout(self,mapping,player_id):
        state=base._parse_state(mapping); neighbors=self.bot._get_neighbors(state); danger,predicted,bomb_until=self.bot._predict(state)
        plan=self.bot._survival_plan(state,player_id,danger,bomb_until,neighbors)
        if self.bot._future_hit(danger,state.players[player_id].row,state.players[player_id].col):
            return np.asarray([plan.first_action,base.ABILITY_NONE],np.int32)
        move=self._fallback_move(mapping,player_id)
        enemy=state.players[1-player_id]; player=state.players[player_id]
        close=abs(player.row-enemy.row)+abs(player.col-enemy.col)<=2
        ability=base.ABILITY_BOMB if close and player.bombs>0 and plan.survived else base.ABILITY_NONE
        return np.asarray([move,ability],np.int32)

def decide_batch(states,player_ids,config):
    bot=TacticalFamilyBot(config)
    return np.stack([bot.decide(state,int(pid)) for state,pid in zip(states,player_ids)]).astype(np.int32)
