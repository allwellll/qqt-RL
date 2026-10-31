"""One CPU owner, no GPU initialization and no training launch path."""
import os,sys,time,subprocess
from pathlib import Path
from scripts.population_run_state import lock,atomic_json
from scripts.combat_mix_owner import read
from scripts.bomb_threat_owner import invalid_ancestry
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/bomb_threat_v1'
BASELINE_ORDER={'baseline_parent':0,'baseline_u1120':1,'baseline_slot01_latest':2}
REFERENCE_ORDER={'baseline_parent_dev':0,'reference_transport1536':1,'reference_selfplay3072':2}
def request_order(p):
    # The three mandated fresh baselines always precede checkpoint follow-ups.
    if p.stem in BASELINE_ORDER:return (0,BASELINE_ORDER[p.stem],p.name)
    if p.stem in REFERENCE_ORDER:return (.5,REFERENCE_ORDER[p.stem],p.name)
    # Global FIFO prevents a continually growing low-number slot from
    # starving other slots. Request files are immutable once published.
    try:created=p.stat().st_mtime
    except OSError:created=0
    return (1,created,p.name)

def active_evaluations():
    """Adopt a still-running CPU child after an owner-only restart."""
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if b'scripts.bomb_threat_eval' not in argv:continue
            if (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            result.append(next(a.decode() for a in argv if a.startswith(b'runs/bomb_threat_v1/eval_requests/')))
        except (OSError,StopIteration):continue
    return result


def cpu_lanes():
    # Avoid SMT siblings too: this host pairs CPU c with c+96. Training
    # uses physical cores 0:16,32:48,72:96; evaluation gets 36 other cores.
    cores=list(range(16,32))+list(range(48,68))
    return [cores[i:i+3] for i in range(0,len(cores),3)]


def available_cpu_lane():
    occupied=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if b'scripts.bomb_threat_eval' not in (proc/'cmdline').read_bytes().split(b'\0'):continue
            occupied.update(os.sched_getaffinity(int(proc.name)))
        except (OSError,ValueError):continue
    for lane in cpu_lanes():
        if not occupied.intersection(lane):return ','.join(map(str,lane))
    return None


def request_kind(req):
    if req.get('split')=='holdout':return 'holdout'
    cp=req.get('checkpoint')
    if not cp or not Path(cp).parent.name.startswith('bt_slot'):return 'reference'
    return 'cadence' if int(Path(cp).name[1:])%40==0 else 'technical_boundary'


def round16_ready(req):
    """08:15 holdouts follow the complete production-JS fresh matrix."""
    dependency=req.get('after_complete')
    if not dependency:return True
    path=Path(dependency)
    if path.is_absolute() or '..' in path.parts:return False
    return (ROOT/path).exists()


def catchup_order(paths,reference_paths=None,active_runs=None):
    records=[(p,read(p,{})) for p in paths];latest={}
    priority=read(BASE/'GATE_AMEND2_HOLDOUT_PRIORITY.json',{}).get('names',[])
    if active_runs is None:
        from scripts.bomb_threat_owner import causal_lineage_limits
        active_runs={f'slot{int(g):02d}':causal_lineage_limits(e['run'],BASE)
            for g,e in read(BASE/'SLOTS.json',{}).items()}
    def retired(r):
        cp=Path(r['checkpoint']);run=cp.parent.name;slot=run.split('_')[1]
        limits=active_runs.get(slot)
        return bool(limits is not None and (run not in limits or int(cp.name[1:])>limits[run]))
    frontier=[(p,read(p,{})) for p in reference_paths] if reference_paths is not None else records
    for p,r in frontier:
        if request_kind(r)!='cadence':continue
        if retired(r):continue
        slot=Path(r['checkpoint']).parent.name.split('_')[1];u=int(Path(r['checkpoint']).name[1:])
        latest[slot]=max(latest.get(slot,0),u)
    def key(pair):
        p,r=pair;kind=request_kind(r)
        if r.get('name')=='fresh256_a3sr_u640':return (-4,0,0,p.name)
        if r.get('round16_priority') and round16_ready(r):return (-3.5,r['round16_priority'],0,p.name)
        if p.stem in priority:return (-3,priority.index(p.stem),0,p.name)
        if p.stem in BASELINE_ORDER:return (-2,BASELINE_ORDER[p.stem],0,p.name)
        if kind in ('holdout','reference'):return (-1,request_order(p)[0],0,p.name)
        u=int(Path(r['checkpoint']).name[1:]);slot=Path(r['checkpoint']).parent.name.split('_')[1]
        if kind=='technical_boundary':return (4,0,-u,p.name)
        if retired(r):return (3,0,slot,p.name)
        distance=latest[slot]-u
        tier=0 if distance==0 else 1 if distance%80==0 else 2
        return (tier,distance,slot,p.name)
    return [p for p,r in sorted(records,key=key)]


def archive_ineligible_requests():
    active=set(active_evaluations());counts={'excluded':0,'technical':0}
    for p in list((BASE/'eval_requests').glob('*.json')):
        if str(p.relative_to(ROOT)) in active:continue
        r=read(p,{});cp=r.get('checkpoint') or '';run=Path(cp).parent.name
        excluded=run.startswith('bt_slot') and (invalid_ancestry(run) or (BASE/'formal'/run/'EXCLUDED_EVALUATION.json').exists())
        technical=request_kind(r)=='technical_boundary'
        if not excluded and not technical:continue
        category='excluded' if excluded else 'technical';dest=BASE/(category+'_requests')/p.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            assert dest.read_bytes()==p.read_bytes();p.unlink()
        else:p.replace(dest)
        atomic_json(BASE/'request_disposition'/p.name,dict(category=category,checkpoint=cp,
            reason='excluded ancestry' if excluded else 'non-40 boundary: technical-only, no full games requested',
            preserved_request=str(dest.relative_to(ROOT)),unix=time.time()))
        counts[category]+=1
    return counts

def refresh_gates():
    from scripts.bomb_threat_gate_report import refresh_all
    try:
        refresh_all()
    except Exception as error:
        # Passive postprocessing must not terminate the queue or relaunch games.
        atomic_json(BASE/'GATE_REPORT_FAILURE.json',dict(status='RETRY_REQUIRED',
            error_type=type(error).__name__,training_unaffected=True,raw_games_preserved=True,unix=time.time()))
        return False
    failure=BASE/'GATE_REPORT_FAILURE.json'
    if failure.exists():
        atomic_json(BASE/'GATE_REPORT_RECOVERY.json',dict(status='RECOVERED',previous=read(failure,{}),unix=time.time()))
        failure.unlink()
    return True

def refresh_metrics():
    """Independent of GPU training; failed uploads remain durably retryable."""
    started=time.time()
    # Consume the authorized passive audit independently from all training.
    audit=BASE/'signal_audit_fullgames_20261029_attempt02/COMPLETE.json'
    consumed=read(BASE/'SIGNAL_AUDIT_CONSUMED.json',{})
    if audit.exists():
        from scripts.population_run_state import sha
        if consumed.get('complete_sha256')!=sha(audit):
            try:
                from scripts.bomb_threat_signal_audit_report import write_report
                report=write_report();status=read(BASE/'SIGNAL_AUDIT_REPORT_STATUS.json',{})
                atomic_json(BASE/'SIGNAL_AUDIT_CONSUMED.json',dict(status=status['status'],complete_sha256=sha(audit),
                    report=str(report.relative_to(ROOT)),report_sha256=sha(report),unix=time.time()))
                if status['fullgame_status']=='COMPLETE':
                    execution=read(BASE/'GATE_AMEND2_EXECUTION.json',{})
                    execution.update(status='COMPLETED_TRAINING_CONTINUES',fullgame_audit_verification=
                        'runs/bomb_threat_v1/signal_audit_fullgames_20261029_attempt02/VERIFICATION.json',unix=time.time())
                    atomic_json(BASE/'GATE_AMEND2_EXECUTION.json',execution)
                    from scripts.iter_pbt_owner import notify
                    notify('BOMB_THREAT_GATE_AMEND2_SIGNAL_COMPLETE',
                        '修订2与完整自然局信号审计已完成，原始trace独立复算通过；训练持续，奖励未改，未晋升BEST。',report=str(report.relative_to(ROOT)))
            except Exception as error:
                atomic_json(BASE/'SIGNAL_AUDIT_REPORT_FAILURE.json',dict(status='RETRY_REQUIRED',
                    error_type=type(error).__name__,training_unaffected=True,unix=time.time()))
    gates_ok=refresh_gates()
    marker=BASE/'METRICS_REFRESH.json'
    atomic_json(marker,dict(status='PENDING',unix=time.time()))
    with (BASE/'logs/METRICS_AUTO.log').open('ab') as log:
        try:
            code=subprocess.run([sys.executable,'-m','scripts.build_model_metrics_table','--upload'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu'),
                stdout=log,stderr=subprocess.STDOUT,timeout=150).returncode
        except subprocess.TimeoutExpired:code=124
    atomic_json(marker,dict(status='VERIFIED' if code==0 and gates_ok else 'RETRY_PENDING',exit_code=code,processed_through_unix=started,unix=time.time()))
def main():
    guard=lock(BASE/'EVAL_OWNER.lock');children={};publisher=None;last_audit=0
    while not (BASE/'USER_STOP_EVAL.json').exists():
        for name,(proc,stream,out,log) in list(children.items()):
            code=proc.poll()
            if code is None:continue
            stream.close();del children[name]
            if code:atomic_json(out/'FAILURE.json',dict(exit_code=code,log=str(log.relative_to(ROOT)),unix=time.time()))
        archive_ineligible_requests()
        active=active_evaluations()
        # Launch the ready GPU-independent work before expensive reports/upload.
        paths=list((BASE/'eval_requests').glob('*.json'))
        unfinished=[];quick=[]
        for p in paths:
            out=BASE/'eval'/p.stem
            if str(p.relative_to(ROOT)) in active or (out/'COMPLETE.json').exists() or (out/'FAILURE.json').exists():continue
            (quick if (out/'NATURAL_COMPLETE.json').exists() else unfinished).append(p)
        ordered=catchup_order(unfinished,reference_paths=paths)+catchup_order(quick,reference_paths=paths)
        for p in ordered:
            if len(active)>=12:break
            affinity=available_cpu_lane()
            if affinity is None:break
            req=read(p,{});out=BASE/'eval'/p.stem;out.mkdir(parents=True,exist_ok=True)
            if req.get('native_only') or not round16_ready(req):continue
            log=BASE/'logs'/f'eval_{p.stem}.log'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                QQT_EVAL_SHARDS='3',QQT_GATE_DEFER='1')
            cmd=['taskset','-c',affinity,sys.executable,'-m','scripts.bomb_threat_eval',str(p.relative_to(ROOT))]
            natural_only=not (out/'NATURAL_COMPLETE.json').exists()
            if natural_only:cmd.append('--natural-only')
            f=log.open('ab',buffering=0)
            proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            children[p.stem]=(proc,f,out,log);active.append(str(p.relative_to(ROOT)))
            atomic_json(out/'EVAL_LAUNCH.json',dict(request=str(p.relative_to(ROOT)),cpu_affinity=affinity,
                natural_first=natural_only,cpu_only=True,unix=time.time()))
        # A unique disposable passive publisher cannot delay filling worker slots.
        if publisher and publisher[0].poll() is not None:publisher[1].close();publisher=None
        refresh=read(BASE/'METRICS_REFRESH.json',{})
        newest=max((p.stat().st_mtime for glob in ('*/NATURAL_COMPLETE.json','*/SUMMARY.json','*/native/NATIVE_COMPLETE.json')
                    for p in (BASE/'eval').glob(glob)),default=0)
        report_due=time.time()>=read(BASE/'ROUND_REPORT_STATUS.json',{}).get('next_due_unix',0)
        if publisher is None and (report_due or newest>refresh.get('processed_through_unix',refresh.get('unix',0)) or refresh.get('status') in ('PENDING','RETRY_PENDING')):
            if time.time()-refresh.get('unix',0)>30:
                stream=(BASE/'logs/catchup_publisher.log').open('ab',buffering=0)
                proc=subprocess.Popen(['taskset','-c','68-71',sys.executable,'-m','scripts.bomb_threat_eval_owner','--publish-once'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT)
                publisher=(proc,stream)
        if time.time()-last_audit>=30:
            from scripts.bomb_threat_catchup_acceptance import refresh
            try:refresh()
            except Exception as error:
                import traceback
                frames=[]
                for frame in traceback.extract_tb(error.__traceback__):
                    path=Path(frame.filename)
                    name=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
                    frames.append(dict(file=name,line=frame.lineno,function=frame.name))
                atomic_json(BASE/'CATCHUP_AUDIT_FAILURE.json',dict(error_type=type(error).__name__,
                    reason=str(error) if '/' not in str(error) else 'path-bearing message omitted; see relative frames',
                    traceback_frames=frames,unix=time.time()))
            last_audit=time.time()
        atomic_json(BASE/'CPU_EVAL_STATUS.json',dict(status='RUNNING',active=active,max_workers=12,
            pending_natural=len(unfinished),pending_quickscreen=len(quick),cpu_lanes=cpu_lanes(),
            frontier_scope='active causal lineage; retired branches retained at lower priority',unix=time.time()))
        time.sleep(2)
if __name__=='__main__':
    if '--publish-once' in sys.argv:
        try:guard=lock(BASE/'METRICS_PUBLISH.lock')
        except BlockingIOError:raise SystemExit(0)
        from scripts.bomb_threat_round_report import write_round
        write_round()
        refresh_metrics()
        from scripts.bomb_threat_catchup_acceptance import refresh
        refresh()
    else:main()
