"""Continue authorized staged cleanup in a separate receipt namespace."""
import json
from pathlib import Path
import time
from scripts import best_model_cleanup_20261030 as C
from scripts.population_run_state import atomic_json, lock, sha

B=Path('runs/bomb_threat_v1')
OLD=B/'publish_best_20261030'
OUT=OLD/'cleanup_after_reboot_20261030'


def configure():
    OUT.mkdir(parents=True,exist_ok=True)
    C.O=C.ROOT/OUT
    C.PLAN=C.O/'CLEANUP_PLAN_20261030.json'
    C.RECEIPT=C.O/'CLEANUP_RECEIPT_20261030.json'
    for name in ('ONLINE_CATALOG_BEFORE.json','SELECTION_PROTOCOL.json'):
        target=OUT/name
        if not target.exists():target.write_bytes((OLD/name).read_bytes())
    required=C.required_pins
    def keep():
        result=required()
        manifest=json.loads((B/'REBOOT_RESUME_MANIFEST_20261030.json').read_text())
        for s in manifest['slots'].values():result[s['checkpoint']]='exact pre-reboot source retained through recovery/publication task'
        return result
    C.required_pins=keep


def main(stage):
    configure()
    guard=lock(OUT/'CLEANUP.lock')
    result=C.build() if stage=='plan' else C.apply(stage)
    if stage=='plan':
        print(json.dumps({k:result[k] for k in ('status','count','bytes')},ensure_ascii=False))
        return
    print(json.dumps(dict(status=result['status'],stage=stage,deleted_exports=len(result['deleted_exports']),revoked_pins=len(result['revoked_pins']))))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
