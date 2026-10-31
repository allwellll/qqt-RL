"""Scheduling-only CPU partitions and conservative training-throughput guard."""
DEFAULT={'gpu':1,'cpu':3,'evaluation':4,'analysis':2,'js':1}
WIDTHS={'cpu':3,'evaluation':3,'analysis':2,'js':2}

def limits(config):
    result=dict(DEFAULT);result.update(config.get('limits',{}))
    if result['gpu']!=1 or any(not isinstance(x,int) or x<0 or x>DEFAULT.get(r,0) for r,x in result.items()):
        raise ValueError('Resource caps exceed authorized CPU partitions or unique GPU owner')
    return result

def scheduling_slot(resource,active,preferred=None):
    if resource=='gpu':return 0
    def free(slot):return f'{resource}:{slot}' not in active
    if isinstance(preferred,int) and 0<=preferred<DEFAULT[resource] and free(preferred):return preferred
    return next(i for i in range(DEFAULT[resource]) if free(i))

def cpu_group(available,resource,slot,offset=12):
    if resource not in WIDTHS or not 0<=slot<DEFAULT[resource]:
        raise ValueError('No authorized CPU partition for this resource slot')
    for lane,count in [('cpu',3),('evaluation',4),('analysis',2),('js',1)]:
        if lane==resource:
            return available[offset+slot*WIDTHS[lane]:offset+(slot+1)*WIDTHS[lane]]
        offset+=count*WIDTHS[lane]
    return []

def regression(reference,samples):
    if not reference or len(samples)<20:return None
    import statistics
    return statistics.median(samples[-20:])<reference*.95

def ramp_ready(reference,samples):
    import statistics
    return bool(reference and len(samples)>=20 and statistics.median(samples[-20:])>=reference*.98)

def row_rate(row):
    """V2 records env/s; SCALE records the same sample count and cycle."""
    if row.get('env_per_second') is not None:return float(row['env_per_second'])
    return float(row['actual_environment_steps_this_update'])/float(row['cycle_seconds'])

def reap_children():
    import os
    count=0
    while True:
        try:identity,_=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:break
        if not identity:break
        count+=1
    return count

def _identity(process):
    return getattr(process,'identity',getattr(process,'pid',None))

def _children(identity):
    from pathlib import Path
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            stat=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[1])==identity:found.append((int(entry.name),stat[0]))
        except OSError:pass
    return found

def request_js_handoff(campaign,value,matching_process):
    import os,signal,time
    job=next(j for j in campaign['jobs'] if j['id']=='PRODUCTION-JS-MATRIX')
    if job['args'][1]=='scripts.campaign_js_parallel':return True
    assert job['args'][1]=='scripts.campaign_js_matrix'
    identity=matching_process(job)
    if identity is None:return False
    os.kill(identity,signal.SIGSTOP)
    campaign['js_scheduler_handoff']=dict(job=job['id'],new_args=value['new_args'],complete=False,requested_unix=time.time())
    return True

def js_handoff_tick(campaign,active,base):
    import os,signal,time
    from pathlib import Path
    from scripts.population_run_state import atomic_json
    handoff=campaign.get('js_scheduler_handoff')
    if not handoff or handoff.get('complete'):return
    for key,(job,process,stream) in list(active.items()):
        if job['id']!=handoff['job']:continue
        identity=_identity(process)
        try:
            state=(Path('/proc')/str(identity)/'stat').read_text().rsplit(')',1)[1].split()[0]
        except OSError:state='gone'
        if state not in ('T','t','Z','gone') and not handoff.get('termination_requested'):return
        # Stopped coordinator cannot claim another batch. Its existing Node
        # child commits RESULT.json before the coordinator is replaced.
        if any(state!='Z' for _,state in _children(identity)):return
        if not handoff.get('termination_requested'):
            try:os.kill(identity,signal.SIGTERM);os.kill(identity,signal.SIGCONT)
            except ProcessLookupError:pass
            handoff['termination_requested']=True
            return
        if process.poll() is None:return
        stream.close();del active[key]
        job.update(args=handoff['new_args'],status='REGISTERED',scheduler_revision='parallel_v1')
        handoff['complete']=True
        atomic_json(base/'JS_PARALLEL_HANDOFF.json',dict(passed=True,old_inflight_batch_committed=True,completed_batches_preserved=True,job=job['id'],unix=time.time()))

def throughput_guard(campaign,active,config,config_path,root,base):
    import os,signal,json,time,statistics
    from scripts.population_run_state import atomic_json
    if not config.get('throughput_guard'):return
    gpu=active.get('gpu')
    if gpu:
        job=gpu[0];path=root/job.get('progress',job['complete'])
        try:
            rows=[json.loads(s) for s in (path.parent/'updates.jsonl').read_text().splitlines() if s.strip()]
            warm=[r for r in rows if r.get('new_update',0)>2]
            calibration=campaign.get('eval_gpu_calibration')
            if calibration is None and '-TECH-' not in job['id'] and config.get('calibrate_same_run'):
                calibration=dict(job=job['id'],phase='low_concurrency',complete=False,started_unix=time.time(),desired_limits=dict(config['limits']),low_start_update=warm[-1]['new_update'] if warm else 2)
                campaign['eval_gpu_calibration']=calibration
                for running,process,_ in active.values():
                    if running['resource'] in ('evaluation','cpu') and running.get('scheduling_slot',0)>0:
                        try:os.killpg(_identity(process),signal.SIGSTOP);running['resource_guard_suspended']=True
                        except ProcessLookupError:pass
                config['limits'].update(evaluation=1,cpu=1);atomic_json(config_path,config)
            if calibration and calibration['job']==job['id'] and not calibration['complete']:
                if calibration['phase']=='low_concurrency':
                    low=[r for r in warm if r['new_update']>calibration['low_start_update']]
                    if len(low)>=20:
                        calibration.update(phase='expanded_concurrency',low_records=low[:20],high_start_update=warm[-1]['new_update'],low_median_env_per_second=statistics.median(row_rate(r) for r in low[:20]),low_median_cycle_seconds=statistics.median(r['cycle_seconds'] for r in low[:20]))
                        config['limits']=calibration['desired_limits'];atomic_json(config_path,config)
                        campaign.setdefault('eval_gpu_references',{})[job['id']]=dict(reference=calibration['low_median_env_per_second'],after_update=calibration['high_start_update'],scope='same-run controlled concurrency; compile excluded')
                else:
                    high=[r for r in warm if r['new_update']>calibration['high_start_update']]
                    if len(high)>=20:
                        speed=statistics.median(row_rate(r) for r in high[:20])
                        calibration.update(complete=True,phase='complete',high_records=high[:20],high_median_env_per_second=speed,high_median_cycle_seconds=statistics.median(r['cycle_seconds'] for r in high[:20]),throughput_change_fraction=speed/calibration['low_median_env_per_second']-1,passed=speed>=calibration['low_median_env_per_second']*.95,completed_unix=time.time())
                        atomic_json(base/'EVAL_GPU_CALIBRATION.json',calibration)
            refs=campaign.setdefault('eval_gpu_references',{})
            calibrating_low=bool(calibration and calibration['job']==job['id'] and calibration['phase']=='low_concurrency')
            if job['id'] not in refs and len(warm)>=20 and not calibrating_low:
                refs[job['id']]=dict(reference=statistics.median(row_rate(r) for r in warm[-20:]),after_update=warm[-1]['new_update'],scope='same-run warm window; compile excluded')
            if job['id'] in refs:
                ref=refs[job['id']];later=[r for r in warm if r['new_update']>ref['after_update']]
                if regression(ref['reference'],[row_rate(r) for r in later]) and not ref.get('reduced'):
                    new_caps={r:max(1,(config['limits'][r]+1)//2) for r in ('evaluation','cpu')}
                    for running,process,_ in active.values():
                        if running['resource'] in new_caps and running.get('scheduling_slot',0)>=new_caps[running['resource']]:
                            try:os.killpg(_identity(process),signal.SIGSTOP);running['resource_guard_suspended']=True
                            except ProcessLookupError:pass
                    config['limits'].update(new_caps);atomic_json(config_path,config)
                    ref['reduced']=True
                    ref['reduction_update']=warm[-1]['new_update']
                    atomic_json(base/'EVAL_GPU_THROUGHPUT_ALERT.json',dict(job=job['id'],reference=ref,observed_median_env_per_second=statistics.median(row_rate(r) for r in later[-20:]),action='suspend surplus CPU observers; retained all state; GPU uninterrupted',unix=time.time()))
                if ref.get('reduced'):
                    ref.setdefault('reduction_update',warm[-1]['new_update'] if warm else 0)
                    recovered=[r for r in warm if r['new_update']>ref['reduction_update']]
                    if len(recovered)>=20 and statistics.median(row_rate(r) for r in recovered[-20:])>=ref['reference']*.98:
                        targets=campaign.get('eval_gpu_calibration',{}).get('desired_limits',DEFAULT)
                        config['limits'].update({r:min(targets[r],config['limits'][r]+1) for r in ('evaluation','cpu')})
                        atomic_json(config_path,config)
                        ref.update(reduced=False,after_update=warm[-1]['new_update'])
                        atomic_json(base/'EVAL_GPU_RECOVERY.json',dict(job=job['id'],limits=config['limits'],warm_recovery_updates=20,unix=time.time()))
                elif not regression(ref['reference'],[row_rate(r) for r in later]):
                    targets=campaign.get('eval_gpu_calibration',{}).get('desired_limits',DEFAULT)
                    if any(config['limits'][r]<targets[r] for r in ('evaluation','cpu')) and ramp_ready(ref['reference'],[row_rate(r) for r in later]):
                        config['limits'].update({r:min(targets[r],config['limits'][r]+1) for r in ('evaluation','cpu')})
                        atomic_json(config_path,config)
                        ref['after_update']=warm[-1]['new_update']
                        atomic_json(base/'EVAL_GPU_RAMP.json',dict(job=job['id'],limits=config['limits'],warm_healthy_updates=20,unix=time.time()))
        except (OSError,ValueError,KeyError):pass
    caps=limits(config)
    for resource in ('cpu','evaluation'):
        live=[x for x in active.values() if x[0]['resource']==resource]
        count=sum(not j.get('resource_guard_suspended') for j,_,_ in live)
        for running,process,_ in live:
            if running.get('resource_guard_suspended') and count<caps[resource]:
                try:os.killpg(_identity(process),signal.SIGCONT);running['resource_guard_suspended']=False;count+=1
                except ProcessLookupError:pass
