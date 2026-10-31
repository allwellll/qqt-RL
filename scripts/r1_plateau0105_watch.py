"""Passive plateau/retirement diagnostics. Never dispatch training or publishing."""
import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from scripts.population_run_state import atomic_json, lock, sha
from scripts.custody_performance2_acceptance import update_rows, verify_file

BASE = Path('runs/bomb_threat_v1')
R2 = Path('runs/bun_native_v2_20261030')
OUT = BASE/'iter2/r1_attack_20261030/plateau_0105_20261031'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def alert_pair(points):
    last = sorted(points, key=lambda r: r['update'])[-2:]
    return (len(last) == 2 and last[1]['update']-last[0]['update'] == 40
            and all(r['metrics']['games'] == 64 and r['metrics']['bombs_per_game'] > 60
                    and r['metrics']['causal_kills'] < 50 for r in last))


def classify(ratio):
    if ratio >= .60:
        return 'CO_TENANCY_BOTTLENECK_OBSERVED'
    if ratio < .40:
        return 'FULL_128_TICK_PROFILE_REQUIRED'
    return 'INCONCLUSIVE_40_TO_60'


def post_exclusive(history, since, cold_boundary):
    warm = [r for r in history if r['new_update'] > cold_boundary][2:]
    # A cycle which began while R1 was still present is never a solo sample.
    return [r for r in warm if r['unix']-r['cycle_seconds'] >= since][-5:]


def snapshot(folder, since=None, boundary=None):
    history = update_rows(folder/'updates.jsonl') if (folder/'updates.jsonl').exists() else []
    if since is None:
        warm = history[2:][-5:]
    else:
        warm = post_exclusive(history, since, boundary)
    if len(warm) < 5:
        return dict(status='WAITING_5_WARM_POINTS',points=[r['new_update'] for r in warm])
    assert all(r['finite'] and min(r['update_l2'].values()) > 0 for r in warm)
    return dict(status='MEASURED',run=folder.name,points=[r['new_update'] for r in warm],
        unix_points=[r['unix'] for r in warm],
        cycle_seconds=statistics.median(r['cycle_seconds'] for r in warm),
        env_per_second=statistics.median(r['actual_environment_steps_this_update']/r['cycle_seconds'] for r in warm))


def notify_once(name, message, evidence):
    path=OUT/'notifications'/f'{name}.json'
    if path.exists():return
    atomic_json(path,dict(message=message,evidence=evidence,unix=time.time()))
    with (OUT/'NOTIFY.log').open('ab') as log:
        result=subprocess.run(['jdme-notify','send',message],stdout=log,stderr=subprocess.STDOUT,check=False)
    atomic_json(path,dict(message=message,evidence=evidence,command_exit_code=result.returncode,unix=time.time()))


def verified_point(path):
    """Recompute64 original full games, with exact input hashes; no new games."""
    d=read(path);cache=OUT/'metric_cache'/f'{path.parent.name}.json'
    inputs={str(p):dict(sha256=d['files'][p.name],size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(path.parent.glob('natural_greedy_[0-9][0-9][0-9].json.outcome.json'))}
    old=read(cache,{})
    if old.get('complete_sha256')==sha(path) and old.get('inputs')==inputs:
        return old
    from scripts.bomb_threat_eval import summarize
    assert len(inputs)==8
    episodes=[]
    for filename,meta in inputs.items():
        p=Path(filename);assert sha(p)==meta['sha256'];raw=read(p)
        assert raw['engine']=='production_JS' and not raw['teacher'] and raw['mode']=='greedy'
        assert raw['simulator_sha256']==d['simulator_sha256'] and len(raw['episodes'])==8
        episodes+=raw['episodes']
    assert len(episodes)==64 and len({r['seed'] for r in episodes})==64
    assert all(r['done'] for r in episodes)
    assert all(sum(r['seat']==s for r in episodes)==32 for s in (0,1))
    metrics=summarize(episodes)
    for key in ('win','loss','draw','causal_kills','actual_bombs','bombs_per_game','zero_bomb_games','self_deaths','cage_kills'):
        assert metrics[key]==d['natural_js_full2400']['greedy'][key],key
    m=re.fullmatch(r'(bt_slot(\d+)_c(\d+))_u(\d+)_holdout',path.parent.name);assert m
    result=dict(run=m[1],slot=int(m[2]),chunk=int(m[3]),update=int(m[4]),metrics=metrics,
        policy_sha256=d['policy_sha256'],source=str(path),complete_sha256=sha(path),inputs=inputs,
        protocol=dict(split=d['split'],seed=d['seed'],spawn_sha256=d['spawn_sha256'],
                      simulator_sha256=d['simulator_sha256'],measurement_version=d['measurement_version'],
                      rule_version=d['rule_version']),training_reward_version=d['reward_version'])
    atomic_json(cache,result);return result


def lineage_limits(current_run):
    """Follow actual checkpoint ancestry, never pick a policy by score/newness."""
    limits={};run=current_run;ceiling=None
    while run not in limits:
        limits[run]=ceiling
        p=read(BASE/'protocols'/f'{run}.json',{})
        source=p.get('continuation_source')
        if not source:
            sources=list(p.get('sources',{}).values())
            if len(sources)!=1:break
            source=sources[0]
        cp=Path(source['checkpoint'])
        if cp.parent.parent!=BASE/'formal':break
        run=cp.parent.name;ceiling=source['update']
    return limits


def metric_points(slot, reward_version, limits):
    candidates=[]
    for path in (BASE/'eval').glob(f'bt_slot{slot:02d}_c*_u*_holdout/NATURAL_COMPLETE.json'):
        d=read(path)
        if d.get('reward_version')!=reward_version or d.get('split')!='holdout':continue
        if d.get('natural_js_full2400',{}).get('greedy',{}).get('games')!=64:continue
        m=re.fullmatch(r'bt_slot\d+_c(\d+)_u(\d+)_holdout',path.parent.name)
        if not m:continue
        run=path.parent.name.rsplit('_u',1)[0]
        if run not in limits:continue
        if limits[run] is not None and int(m[2])>limits[run]:continue
        if m and int(m[2])%40==0:candidates.append((int(m[2]),int(m[1]),path))
    latest={}
    for update,chunk,path in sorted(candidates):
        if update in latest:
            assert read(path)['policy_sha256']==read(latest[update])['policy_sha256'],'ambiguous same-update policies'
        latest[update]=path
    points=[verified_point(path) for _,path in sorted(latest.items())[-8:]]
    if points:
        assert all(r['protocol']==points[-1]['protocol'] for r in points),'mixed evaluation protocols'
    return points


def gpu_apps(gpu):
    """Identify this card's apps in memory; persist neither PID nor device UUID."""
    cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid=next(r[1].strip() for r in csv.reader(cards.splitlines()) if int(r[0])==gpu)
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    apps=[]
    for row in csv.reader(raw.splitlines()):
        if row[0].strip()!=uuid:continue
        try:args=[x.decode() for x in Path('/proc',row[1].strip(),'cmdline').read_bytes().split(b'\0') if x]
        except FileNotFoundError:return ['PROCESS_TRANSITION_RETRY']
        if '-m' not in args:apps.append('UNIDENTIFIED_COMPUTE_APP');continue
        at=args.index('-m');module=args[at+1]
        run=args[at+3] if len(args)>at+3 and args[at+2].startswith('runs/') else 'UNKNOWN_RUN'
        apps.append(module+':'+run)
    return sorted(apps)


def retirement_diagnostic(slots):
    saved=read(OUT/'POST_RETIREMENT_COMPARISON.json')
    if saved and saved.get('slot07_cold_excluded_updates') and 'c_fraction_of_a' in saved:
        return saved
    if saved and saved.get('status')!='SUPERSEDED_NOT_FORMAL_COTENANT':
        atomic_json(OUT/'POST_RETIREMENT_PRE_FORMAL_SNAPSHOT.json',saved)
        atomic_json(OUT/'POST_RETIREMENT_COMPARISON.json',dict(
            status='SUPERSEDED_NOT_FORMAL_COTENANT',
            reason='previous samples preceded slot07 formal training; not a shared-c comparison',
            preserved='POST_RETIREMENT_PRE_FORMAL_SNAPSHOT.json',unix=time.time()))
    request=read(BASE/'iter2/r1_attack_20261030/stop2255/slot04_STOPPED.json',{})
    if request.get('status')!='COMPLETE_CHUNK_THEN_RETIRED':
        return dict(status='WAITING_SLOT04_U4240_RETIREMENT',run=slots['4']['run'])
    done=request['last_atomic'];assert done['update']==4240 and done['passed']
    cp=Path(done['checkpoint'])
    verify_file(cp/'state.pkl',done['state_sha256']);verify_file(cp/'policy.pt',done['policy_sha256'])
    r2=read(R2/'R2_SLOTS.json');apps4=gpu_apps(4);apps7=gpu_apps(7)
    expected=r2['b'].get('worker_module','scripts.bun_native_v2_train')+':'+r2['b']['run']
    before=read(OUT/'EXCLUSIVITY.json',{})
    if apps4!=[expected]:
        atomic_json(OUT/'EXCLUSIVITY.json',dict(status='NOT_EXCLUSIVE',gpu4_apps=apps4,unix=time.time()))
        return dict(status='WAITING_GPU4_R2B_ONLY',gpu4_apps=apps4)
    if before.get('status')!='EXCLUSIVE':
        before=dict(status='EXCLUSIVE',since=time.time(),gpu4_apps=apps4,slot04_terminal=request)
        atomic_json(OUT/'EXCLUSIVITY.json',before)
    launch=read(BASE/'formal'/slots['7']['run']/'LAUNCH.json',{})
    if launch.get('stage')!='formal':
        return dict(status='WAITING_R1_SLOT07_FORMAL_COTENANT',gpu7_apps=apps7,
                    slot07_stage=launch.get('stage'))
    r1_rows=update_rows(BASE/'formal'/slots['7']['run']/'updates.jsonl')
    formal_rows=[r for r in r1_rows if r['unix']>=launch['unix']]
    if len(formal_rows)<2:
        return dict(status='WAITING_SLOT07_TWO_FORMAL_UPDATES',
                    formal_updates=[r['new_update'] for r in formal_rows],gpu7_apps=apps7)
    since=max(before['since'],formal_rows[1]['unix']);measured={}
    for arm,entry in r2.items():
        protocol=read(Path(entry['protocol']));folder=R2/'formal'/entry['run']
        measured[arm]=snapshot(folder,since,protocol['chunk_start_update'])
    if any(x['status']!='MEASURED' for x in measured.values()):
        return dict(status='COLLECTING_POST_RETIREMENT_WARM_POINTS',
                    exclusive_since=before['since'],comparison_since=since,
                    measurements=measured,gpu7_apps=apps7)
    c=r2['c'].get('worker_module','scripts.bun_native_v2_train')+':'+r2['c']['run']
    r1='scripts.bomb_threat_r1_continue_train:'+slots['7']['run']
    if apps7!=sorted([c,r1]):return dict(status='WAITING_COMPARABLE_GPU7_CO_TENANTS',gpu7_apps=apps7)
    ratio=measured['b']['env_per_second']/measured['a']['env_per_second'];decision=classify(ratio)
    result=dict(status=decision,exclusive_since=before['since'],comparison_since=since,
        slot07_formal_launch=launch,measurements=measured,b_fraction_of_a=ratio,
        slot07_cold_excluded_updates=[r['new_update'] for r in formal_rows[:2]],
        c_fraction_of_a=measured['c']['env_per_second']/measured['a']['env_per_second'],
        gpu4_apps=apps4,gpu7_apps=apps7,training_configuration_changed=False,unix=time.time())
    atomic_json(OUT/'POST_RETIREMENT_COMPARISON.json',result)
    message=f"R1 slot04 u4240退休后，R2-b独占5暖态 {measured['b']['env_per_second']:.0f} env/s，为a的{ratio:.1%}；GPU7共卡c {measured['c']['env_per_second']:.0f} env/s。判定{decision}，未改微核/奖励/网络/batch/seed，等待编排方调度决定。"
    if decision=='FULL_128_TICK_PROFILE_REQUIRED':
        # Existing rows give collector/learner totals, but cannot identify Phi's
        # nested GPU wall time. No fabricated subtraction or microbenchmark proxy.
        atomic_json(OUT/'FULL_128_TICK_PROFILE_REQUEST.json',dict(status='BLOCKED_NO_LIVE_PHI_TRACE',
            ticks=128,reason='nsys/ncu unavailable; existing live worker does not expose per-Phi trace or a profiler server; aggregate logs cannot provide nested GPU attribution',
            allowed='one full128tick update; collector/Phi/learner fractions; no optimization change',
            not_completed=True,source=result,unix=time.time()))
        message+=' 技术阻塞：现有日志不能分离Φ耗时且无可附加GPU profiler；已落盘阻塞请求，不伪报profile完成。'
    notify_once('POST_RETIREMENT_FORMAL_WARM_V2_'+decision,message,result)
    return result


def refresh():
    protocol=read(OUT/'PROTOCOL.json');assert sha(Path(protocol['instruction']))==protocol['instruction_sha256']
    slots=read(BASE/'SLOTS.json');points={};summary={}
    for slot in (0,4,7,5):
        p=read(BASE/'protocols'/f"{slots[str(slot)]['run']}.json")
        points[str(slot)]=metric_points(slot,p['reward_version'],lineage_limits(slots[str(slot)]['run']))
        rr=points[str(slot)]
        if rr:
            summary[str(slot)]=dict(points=[dict(run=r['run'],update=r['update'],source=r['source']) for r in rr],
                games_per_point=64,win_minus_loss=statistics.mean(r['metrics']['win_minus_loss'] for r in rr),
                causal_kills=statistics.mean(r['metrics']['causal_kills'] for r in rr),
                bombs_per_game=statistics.mean(r['metrics']['bombs_per_game'] for r in rr),
                zero_bomb_games=statistics.mean(r['metrics']['zero_bomb_games'] for r in rr),
                draw_fraction=statistics.mean(r['metrics']['draw_fraction'] for r in rr))
    if alert_pair(points['0']):
        pair=points['0'][-2:]
        notify_once('SLOT00_HIGH_BOMB_LOW_KILL_'+str(pair[-1]['update']),
            'R1 slot00连续两个holdout greedy64局点：泡/局>60且因果击杀<50；已复算原始outcome并写告警，未调权重/停训/PBT。',pair)
    retirement=retirement_diagnostic(slots)
    result=dict(status='RUNNING_PASSIVE',PBT_copy_allowed=False,publish_allowed=False,deletion_filename_only='state.pkl',
        summary=summary,slot00_latest_points=points['0'][-2:],slot00_alert=alert_pair(points['0']),
        post_retirement_diagnostic=retirement,unix=time.time())
    atomic_json(OUT/'EXECUTION_STATUS.json',result);return result


def reboot_requested():
    return any((base/'STOP_FOR_REBOOT.json').exists() for base in (BASE,Path('runs/iter_pbt_v1')))


def watchdog():
    guard=lock(OUT/'WATCHDOG.lock')
    while not reboot_requested():
        live=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:args=(proc/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if b'scripts.r1_plateau0105_watch' in args and b'--watchdog' not in args and b'--once' not in args:
                live.append(proc)
        assert len(live)<=1,'duplicate passive observers'
        blocked=read(OUT/'BLOCKED.json',{}).get('status')=='BLOCKED_PASSIVE_MONITOR'
        if not live and not blocked:
            with (OUT/'OBSERVER.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m','scripts.r1_plateau0105_watch'],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(OUT/'WATCHDOG_RESTART.json',dict(status='PASSIVE_OBSERVER_RESTARTED',training_untouched=True,unix=time.time()))
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--once',action='store_true');parser.add_argument('--watchdog',action='store_true');args=parser.parse_args()
    if args.watchdog:return watchdog()
    guard=lock(OUT/'OBSERVER.lock')
    while not reboot_requested():
        try:
            refresh()
        except Exception as error:
            failure=dict(status='BLOCKED_PASSIVE_MONITOR',error_type=type(error).__name__,reason=str(error),training_untouched=True,unix=time.time())
            atomic_json(OUT/'BLOCKED.json',failure)
            notify_once('PASSIVE_MONITOR_BLOCKED','R1 plateau只读监控技术阻塞，训练未动；见plateau_0105_20261031/BLOCKED.json。',failure)
            raise
        if args.once:return
        time.sleep(30)


if __name__=='__main__':main()
