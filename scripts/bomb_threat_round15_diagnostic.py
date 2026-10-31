"""CPU-only reproducible re-observation of every round15 unilateral-zero game."""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from scripts.population_run_state import atomic_json, sha, lock

BASE=Path('runs/bomb_threat_v1')
OUT=BASE/'iter2/a3sr/round15/diagnostic'


def read(path):
    return json.loads(Path(path).read_text())


def freeze():
    path=OUT/'MANIFEST.json'
    if path.exists():return read(path)
    proto=read(BASE/'fresh256_a3sr_u640/PROTOCOL.json')
    games=[];files={}
    for group,source,tier,spawn_path,total in [
        ('fresh_hard',BASE/'fresh256_a3sr_u640/hard/candidate','hard',BASE/'fresh256_a3sr_u640/hard_spawns.json',256),
        ('holdout_normal',BASE/'eval/bt_slot05_c000026_u000640_holdout','normal',BASE/'holdout_spawns.json',64)]:
        spawn=read(spawn_path);rows=[]
        pattern='greedy_*.json.outcome.json' if total==256 else 'natural_greedy_*.json.outcome.json'
        for p in sorted(source.glob(pattern)):
            files[str(p)]=sha(p);rows.extend(read(p)['episodes'])
        assert len(rows)==total and len({r['seed'] for r in rows})==total
        for row in rows:
            p=row['seat']
            if row['placed'][p]!=0 or row['placed'][1-p]==0:continue
            i=row['seed']-spawn['seed'];g=i%len(spawn['spawn_cells'])
            assert p==i%2
            games.append(dict(id=len(games),group=group,expected=row,seat=p,
                tier=tier,games=total,episode_index=i,seed_start=spawn['seed'],
                geometry_id=g,birth_cells=spawn['spawn_cells'][g],spawn_manifest=str(spawn_path)))
    assert sum(r['group']=='fresh_hard' and r['seat']==1 for r in games)==45
    assert sum(r['group']=='fresh_hard' and r['seat']==0 for r in games)==10
    doc=dict(id='ROUND15-P1ZERO-20261030',model=proto['models']['candidate'],games=games,
        regions='runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',
        files=files,source_sha256={f:sha(f) for f in ['scripts/bomb_threat_round15_record.js',
            'scripts/bomb_threat_fresh256_eval.js','scripts/bomb_threat_eval.js','web/sim-stock-window-v1.js',
            'scripts/expert_validation_js_eval.js']},cpu_workers=2,cpu_affinity='68-69',teacher=False,
        limitations=['holdout opponent normal differs from fresh hard; not a seed-only comparison',
            '64 geometry IDs reused4 times; seat is fixed by geometry parity, not randomized within each geometry',
            'passive replay of existing games is not new independent evaluation'],unix=time.time())
    atomic_json(path,doc);return doc


def main():
    guard=lock(OUT/'OWNER.lock');doc=freeze()
    for f,h in doc['source_sha256'].items():assert sha(f)==h
    for f,h in doc['files'].items():assert sha(f)==h
    def run(entry):
        out=OUT/f"game_{entry['id']:03d}";accepted=out/'ACCEPTED.json'
        if accepted.exists():
            for f,h in read(accepted)['files'].items():assert sha(f)==h
            return
        out.mkdir(parents=True,exist_ok=True)
        with (out/'LOG.txt').open('ab') as log:
            subprocess.run(['node','scripts/bomb_threat_round15_record.js',str(entry['id'])],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=read(out/'METRICS.json');assert metrics['status']=='MATCH'
        atomic_json(accepted,dict(status='PASS',files={str(f):sha(f) for f in
            [out/'METRICS.json',out/'TRACE.json',out/'evaluation.json.outcome.json']},unix=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=doc['cpu_workers']) as pool:
            fs={pool.submit(run,e):e for e in doc['games']}
            for n,f in enumerate(as_completed(fs),1):
                f.result();atomic_json(OUT/'PROGRESS.json',dict(completed=n,total=len(fs),last=fs[f]['id'],unix=time.time()))
        atomic_json(OUT/'COMPLETE.json',dict(status='PASS',games=len(doc['games']),
            manifest_sha256=sha(OUT/'MANIFEST.json'),all_passive_replays_matched=True,unix=time.time()))
    except Exception as error:
        atomic_json(OUT/'BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,
            detail=str(error),training_untouched=True,unix=time.time()))
        raise


if __name__=='__main__':main()
