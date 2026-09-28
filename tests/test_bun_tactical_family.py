import hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
from jax_bomb.bun_rule_bot import BunRuleBot
from jax_bomb.bun_tactical_family import TacticalFamilyBot,TacticalFamilyConfig

ROOT=Path(__file__).resolve().parents[1]

def state():
 wall=np.zeros((13,15),bool);wall[[0,-1],:]=True;wall[:,[0,-1]]=True
 return {'height':13,'width':15,'wall':wall,'brick':np.zeros_like(wall),'players':[{'row':6,'col':5,'alive':True,'bombs':2,'blast':3,'speed':1.3},{'row':6,'col':7,'alive':True,'bombs':2,'blast':3,'speed':1.3}],'bombs':[]}

def test_full_family_is_exact_v2():
 s=state(); assert np.array_equal(TacticalFamilyBot(TacticalFamilyConfig()).decide(s,0),BunRuleBot().decide(s,0))

def test_family_parameters_change_identity_and_behavior():
 s=state(); full=TacticalFamilyConfig(); easy=TacticalFamilyConfig(name='easy',horizon=24,aggression=0.0,escape_margin=2,tie_break=2,bomb_threshold=1,positioning_preference='space')
 assert full.identity('a','b')!=easy.identity('a','b')
 assert not np.array_equal(TacticalFamilyBot(full).decide(s,0),TacticalFamilyBot(easy).decide(s,0))

def test_heldout_is_structurally_distinct():
 cfg=TacticalFamilyConfig(name='heldout',structural_mode='heldout_evasive_counter',positioning_preference='flank',tie_break=1)
 assert TacticalFamilyBot(cfg).decide(state(),0).shape==(2,)
 assert cfg.structural_mode!='v2'
