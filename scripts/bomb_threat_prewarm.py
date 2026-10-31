"""Two-update isolated warmup then fresh-process restore during ongoing training."""
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.population_run_state import lock,sha,atomic_json
from scripts.bomb_threat_schedule import prewarm_fingerprint,valid_prewarm
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=Path(sys.argv[1]);config=json.loads(protocol.read_text());arm=next(iter(config['arms']));seed=next(iter(config['sources']))
    out=ROOT/config['namespace']/arm;out.mkdir(parents=True,exist_ok=True);guard=lock(out/'PREWARM.lock')
    if (out/'ACCEPTANCE.json').exists():return
    started=time.time()
    for flag,marker in (('--smoke','SMOKE.json'),('--verify-restore','RESTORE.json')):
        if (out/marker).exists():continue
        with (out/(marker+'.log')).open('ab',buffering=0) as log:
            module=config.get('worker_module','scripts.bomb_threat_long_train')
            code=subprocess.run([sys.executable,'-m',module,str(protocol),arm,str(seed),flag],
                env=os.environ,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            atomic_json(out/'FAILURE.json',dict(exit_code=code,stage=flag,unix=time.time()));raise SystemExit(code)
    smoke=json.loads((out/'SMOKE.json').read_text());restore=json.loads((out/'RESTORE.json').read_text())
    assert valid_prewarm(smoke,restore)
    atomic_json(out/'ACCEPTANCE.json',dict(status='PASS',fingerprint=prewarm_fingerprint(config),
        smoke_sha256=sha(out/'SMOKE.json'),restore_sha256=sha(out/'RESTORE.json'),output=str(out.relative_to(ROOT)),
        source_checkpoint=next(iter(config['sources'].values()))['checkpoint'],
        formal_training_steps=0,technical_updates=2,never_evaluate_or_exploit=True,
        elapsed_seconds=time.time()-started,unix=time.time()))

if __name__=='__main__':main()
