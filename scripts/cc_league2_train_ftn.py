"""CC_LEAGUE_V2 trainer with the fountain/chain reward variant (L2_FTN). Same as cc_league2_train plus
  cc_fountain_reward_v1.install_on_import(): r3_items_reward is patched when the worker itself imports it in setup()
  (after XLA_FLAGS). The variant adds no aux key (pressure_count packs total + 64*streak), so no migration.
usage: model_training.py <config> args=[protocol, arm, seed, (--smoke|--verify-restore)]
"""
import json, sys
from pathlib import Path


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())['arms'][sys.argv[2]]
    assert spec.get('reward_variant') == 'cc-fountain-reward-v1', spec.get('reward_variant')
    from scripts import cc_fountain_reward_v1 as V
    V.install_on_import()
    from scripts import cc_league2_train
    cc_league2_train.main()

if __name__ == '__main__':
    main()
