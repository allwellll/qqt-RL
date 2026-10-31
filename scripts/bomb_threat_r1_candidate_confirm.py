"""CPU-only frozen slot00/u2400 followup; no BEST/publication operations."""
import json,time,os,sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.population_run_state import atomic_json,sha,lock
from scripts.bomb_threat_round15 import read
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1';WATCH=BASE/'iter2/a3sr/lineage_watch_20261030';OUT=WATCH/'candidate_slot00_u2400_1440'
MODULE='scripts.bomb_threat_r1_candidate_confirm'

def freeze():
    out=OUT/'PROTOCOL.json'
    if out.exists():return read(out)
    previous=read(WATCH/'PROTOCOL.json');candidate=read(WATCH/'models/slot00_u002400.json');baseline=previous['fresh']['baseline'];models={'candidate':candidate,'baseline':baseline}
    tiers={}
    for tier,seed in [('hard',2026193000),('normal',2026194000)]:
        x=read(ROOT/previous['fresh']['tiers'][tier]['spawn_manifest']);x.update(seed=seed)
        p=OUT/f'{tier}_spawns.json';atomic_json(p,x)
        tiers[tier]=dict(seed=seed,games=256,seats={'P0':128,'P1':128},spawn_manifest=str(p.relative_to(ROOT)),spawn_sha256=sha(p))
    proto=dict(status='FROZEN',instruction='.hermes/STEER_r1_confound_20261030_1440.md',instruction_sha256=sha(ROOT/'.hermes/STEER_r1_confound_20261030_1440.md'),models=models,tiers=tiers,
        mode='greedy',engine='production_JS',rule_version='stock-window-v1',teacher=False,budget_games=1024,
        prerequisite='old slot00/u2400 all3 tiers256 complete; P1 sum>91 and each>=u1280-5',
        limitations=['frozen geometry manifest reused; new seed streams, not256 new unique maps','candidate chosen by declared gate, no publication or BEST'],unix=time.time())
    atomic_json(out,proto);return proto

def gate():
    x=read(WATCH/'RESULTS.json',{});tiers=x.get('fresh',{}).get('0',{});base=x.get('baseline',{})
    if any(tiers.get(t,{}).get('games')!=256 for t in ('hard','normal','easy')):return dict(status='WAITING_EASY_COMPLETE')
    scores={t:tiers[t]['by_seat']['P1']['win_minus_loss'] for t in tiers};bounds={t:base[t]['by_seat']['P1']['win_minus_loss']-5 for t in tiers}
    ok=sum(scores.values())>91 and all(scores[t]>=bounds[t] for t in scores)
    return dict(status='PASS' if ok else 'FAIL',P1_scores=scores,P1_sum=sum(scores.values()),threshold_strict=91,minimum_per_tier=bounds)

def ensure_running():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():return
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if MODULE.encode() in (p/'cmdline').read_bytes().split(b'\0') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return
        except OSError:pass
    log=(OUT/'OWNER.log').open('ab',buffering=0)
    subprocess.Popen(['taskset','-c','124-127',sys.executable,'-m',MODULE],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close()

def main():
    from scripts import bomb_threat_a3sr_lineage_watch as L
    from scripts.iter_pbt_owner import notify
    guard=lock(OUT/'OWNER.lock');proto=freeze();pool=ThreadPoolExecutor(max_workers=4);futures={};L.OUT=OUT
    jobs=[dict(kind='paired_fresh',slot=0 if label=='candidate' else 5,update=2400 if label=='candidate' else 1280,tier=tier,start=i,games=256,seed=t['seed'],spawn_manifest=t['spawn_manifest'],label=label) for label in proto['models'] for tier,t in proto['tiers'].items() for i in range(0,256,8)]
    while True:
        decision=gate();atomic_json(OUT/'GATE.json',decision)
        if decision['status']=='FAIL':atomic_json(OUT/'COMPLETE.json',dict(status='NOT_QUALIFIED_NO_EXTRA_GAMES',gate=decision));return
        if decision['status']!='PASS':time.sleep(10);continue
        if not (OUT/'STARTED.json').exists():
            atomic_json(OUT/'STARTED.json',dict(status='RUNNING',gate=decision,protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time()))
            notify('R1_SLOT00_2400_CONFIRM_START','旧A3SR slot00 u2400 fresh三档P1合计+145，达标；启动hard/normal全新固定seed256局，与u1280同seed各跑，CPU-only，不发布。')
        for f,j in list(futures.items()):
            if f.done():f.result();del futures[f]
        for job in jobs:
            if len(futures)>=4:break
            if job in futures.values() or Path(str(L.shard_path(job))+'.ACCEPTED.json').exists():continue
            futures[pool.submit(L.collect,job,proto['models'][job['label']],proto)]=job
        results={label:{tier:L.summarize_group([j for j in jobs if j['label']==label and j['tier']==tier],model) for tier in proto['tiers']} for label,model in proto['models'].items()}
        n=sum(v['games'] for t in results.values() for v in t.values());status='COMPLETE' if n==1024 else 'RUNNING'
        atomic_json(OUT/'RESULTS.json',dict(status=status,games=n,budget=1024,results=results,protocol_sha256=sha(OUT/'PROTOCOL.json'),unix=time.time()))
        atomic_json(OUT/'EXECUTION_STATUS.json',dict(status=status,games=n,active=list(futures.values()),unix=time.time()))
        if status=='COMPLETE':
            atomic_json(OUT/'COMPLETE.json',dict(status='COMPLETE',games=1024,results_sha256=sha(OUT/'RESULTS.json'),unix=time.time()))
            text=['','## 14:40 旧slot00 u2400新seed复核','', '预先冻结hard 2026193000–3255、normal 2026194000–4255；候选及u1280各512局，全部生产JS greedy，teacher关闭。','']
            for label,tiers in results.items():
                for tier,v in tiers.items():text.append(f"- {label} {tier}: P0 W−L {v['by_seat']['P0']['win_minus_loss']:+d}；P1 {v['by_seat']['P1']['win_minus_loss']:+d}；games={v['games']}。")
            text+=['','原始逐局文件/hash及限制见 `'+str((OUT/'RESULTS.json').relative_to(ROOT))+'`。不改网页/BEST，不将dev复核冒称独立晋升holdout。']
            report=ROOT/'reports/bomb_threat_v1_a3sr_lineage_watch_20261030.md'
            # Separate addendum avoids overwriting the still-running original watcher.
            (ROOT/'reports/bomb_threat_v1_a3sr_candidate_confirm_20261030.md').write_text('\n'.join(text)+'\n')
            notify('R1_SLOT00_2400_CONFIRM_COMPLETE','slot00 u2400 vs u1280同新seed hard/normal共1024局复核完成，结果/hash已落盘，不发布。');return
        time.sleep(5)

if __name__=='__main__':main()
