import numpy as np
from jax_bomb.bun_tactical_labels import derive_action_labels, label_mapping


def _mapping(trap=False):
 wall=np.zeros((13,15),bool);wall[[0,-1],:]=True;wall[:,[0,-1]]=True
 if trap:
  wall[5,7]=wall[7,7]=wall[6,8]=True
 return {'height':13,'width':15,'wall':wall,'brick':np.zeros_like(wall),'players':[{'row':6,'col':5,'alive':True,'bombs':2,'blast':3,'speed':1.3},{'row':6,'col':7,'alive':True,'bombs':2,'blast':3,'speed':1.3}],'bombs':[]}

def test_state_labels_require_self_escape_for_safe_attack():
 label=label_mapping(_mapping(trap=True),0)
 assert label.safe_escape_exists
 assert label.enemy_escape_count >= 0
 assert isinstance(label.safe_attack_available,bool)

def test_action_labels_define_forced_kill_and_trade_separately():
 labels=derive_action_labels(legal=[1,1],survivable=[1,1],abilities=[1,1],self_alive=[1,0],enemy_alive=[0,0],nontrade_kill=[1,0],trade=[0,1],own_bomb_first=[0,1],enemy_escape_count=[0,0],minimum_escape_time=[3,40],causal_kill_onset=[7,-1])
 assert labels['bomb_creates_forced_kill'].tolist()==[True,False]
 assert labels['bomb_creates_trade'].tolist()==[False,True]
 assert labels['own_bomb_death_risk'].tolist()==[False,True]
 assert labels['causal_kill_onset'].tolist()==[7,-1]
 assert labels['forced_kill_onset'].tolist()==[1,-1]


def test_enemy_caused_death_is_not_own_bomb_risk():
 labels=derive_action_labels(legal=[1],survivable=[0],abilities=[0],self_alive=[0],enemy_alive=[1],nontrade_kill=[0],trade=[0],own_bomb_first=[0],enemy_escape_count=[1],minimum_escape_time=[40],causal_kill_onset=[-1])
 assert labels['own_bomb_death_risk'].tolist()==[False]
