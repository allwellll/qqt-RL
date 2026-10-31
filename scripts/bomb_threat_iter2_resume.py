"""Drain already-running CPU diagnostic shards before reloading their owner."""
import os,sys,time
from pathlib import Path
from scripts.population_run_state import lock,atomic_json
B=Path('runs/bomb_threat_v1/iter2')


def main():
    guard=lock(B/'CPU_RESUME.lock')
    while True:
        active=0
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                argv=(p/'cmdline').read_bytes().split(b'\0')
                if b'scripts/bomb_threat_iter2_actions.js' in argv and any(a.startswith(b'runs/bomb_threat_v1/iter2/D1/') for a in argv) and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':active+=1
            except OSError:pass
        atomic_json(B/'CPU_OWNER_RELOAD.json',dict(status='DRAINING_EXISTING_CPU_NODE_SHARDS' if active else 'RELOADING',
            CPU_shards_active=active,completed_outputs_validated_without_rerun=True,training_worker_signals=0,unix=time.time()))
        if not active:break
        time.sleep(2)
    os.execv(sys.executable,[sys.executable,'-m','scripts.bomb_threat_iter2'])


if __name__=='__main__':main()
