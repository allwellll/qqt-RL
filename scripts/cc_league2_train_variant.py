"""CC_LEAGUE_V2 trainer for reward variants, with the import hook that survives model_cuda_overlay (cc_reward_import_hook).
Replaces cc_league2_train_atk / cc_league2_train_ftn, whose install_on_import() finder was shadowed by the overlay
finder (variant never applied). Writes <arm dir>/REWARD_RECEIPT_<pid>.json when the patch is applied.
usage: model_training.py <config> args=[protocol, arm, seed, (--smoke|--verify-restore)]
"""
import importlib, json, os, sys
from pathlib import Path

VARIANTS = {'cc-attack-reward-v2': 'scripts.cc_attack_reward_v2', 'cc-fountain-reward-v1': 'scripts.cc_fountain_reward_v1'}


def main():
    protocol = json.loads(Path(sys.argv[1]).read_text()); arm = sys.argv[2]
    V = importlib.import_module(VARIANTS[protocol['arms'][arm]['reward_variant']])
    os.environ['CC_REWARD_RECEIPT'] = str(Path(protocol['namespace']) / arm / f'REWARD_RECEIPT_{os.getpid()}.json')
    from scripts.cc_reward_import_hook import hook
    hook(V)
    from scripts import cc_league2_train
    cc_league2_train.main()

if __name__ == '__main__':
    main()
