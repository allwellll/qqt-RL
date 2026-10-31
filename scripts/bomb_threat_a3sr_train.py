"""A3SR reset-only worker based on frozen physical-seat A3S optimizer."""
from pathlib import Path
from types import ModuleType


def worker_source():
    source=Path('scripts/bomb_threat_seat5050_train.py').read_text()
    changes={
        'bomb_threat_seat5050_collector as M':'bomb_threat_a3sr_collector as M',
        "BASE/'a3s/cache'":"BASE/'iter2/a3sr/cache'",
        "spec['iter2_arm']=='A3S'":"spec['iter2_arm'] in ('A3S','A3SR')",
        'int(protocol[\'chunk_start_update\'])+1) if args.smoke':'int(protocol[\'chunk_start_update\'])+2) if args.smoke',
        "    if args.smoke:atomic_json(out/'SMOKE.json'":"""    if args.smoke:
        required=('spawn_mode','brick_drop_rate','spawn_resample','spawn_fallback','spawn_eligible')
        assert all(k in row['exposure'] for k in required),'missing reset exposure'
        assert batch_size/row['cycle_seconds']>=18000,'A3SR smoke below18000 env/s'
        assert row['exposure']['spawn_eligible']>0,'no eligible real resets in smoke'
    if args.smoke:atomic_json(out/'SMOKE.json'"""
    }
    for old,new in changes.items():
        assert old in source,old
        source=source.replace(old,new)
    return source


def main():
    module=ModuleType('a3sr_training_worker');module.__file__=str(Path(__file__))
    exec(compile(worker_source(),'scripts/bomb_threat_a3sr_train.py','exec'),module.__dict__)
    module.main()
if __name__=='__main__':main()
