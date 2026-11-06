"""CC_LEAGUE_V2 trainer with the attack reward variant (L2_ATK). Same as cc_league2_train plus
  cc_attack_reward_v2.install() right after the CUDA overlay (reward modules must import after it).
  v2 only rescales existing r3 attack slots, so the aux/ledger structure is unchanged (no migration).
usage: model_training.py <config> args=[protocol, arm, seed, (--smoke|--verify-restore)]
"""
import json, sys
from pathlib import Path


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())['arms'][sys.argv[2]]
    assert spec.get('reward_variant') == 'cc-attack-reward-v2', spec.get('reward_variant')
    # Patch r3_items_reward when the worker imports it in setup() (after XLA_FLAGS); importing it here would
    # initialise the JAX backend early and break deterministic restore (11:50 root cause).
    from scripts import cc_attack_reward_v2 as V
    V.install_on_import()
    from scripts import cc_league2_train
    cc_league2_train.main()

if __name__ == '__main__':
    main()
