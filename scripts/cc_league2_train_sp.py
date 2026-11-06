"""CC_LEAGUE_V2 high self-play trainer (L2_SP): cc_league2_train with cc_league2_sp_lib in place of cc_league2_lib
(gated, history-extended, PFSP G ring). Same reward. Protocol arm field league_lib must be 'cc_league2_sp_lib'.
usage: model_training.py <config> args=[protocol, arm, seed, (--smoke|--verify-restore)]"""
import json, os, sys
from pathlib import Path


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())['arms'][sys.argv[2]]
    assert spec.get('league_lib') == 'cc_league2_sp_lib', spec.get('league_lib')
    os.environ['CC_LEAGUE2_SEED'] = sys.argv[3]
    # cc_league2_lib binds its snapshot BASE at import: set the per-arm root first (same value cc_league2_train sets).
    os.environ['CC_SNAPSHOT_ROOT'] = f'runs/cc_train_20261105/training/league/{sys.argv[2]}'
    import scripts, scripts.cc_league2_sp_lib as sp
    sys.modules['scripts.cc_league2_lib'] = sp; scripts.cc_league2_lib = sp
    from scripts import cc_league2_train
    cc_league2_train.main()


if __name__ == '__main__':
    main()
