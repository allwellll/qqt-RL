"""Complete local/remote playback acceptance and authorized replay-only delivery."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json
import subprocess
import sys
import time
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.best_publish_release_20261030 import upload_replay
from scripts.new_model_replays_20261031 import selected

B=Path('runs/new_model_replays_20261031')
def read(p):return json.loads(Path(p).read_text())
def run(args,log,env=None):
    with (B/log).open('w') as f:
        subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    guard=lock(B/'FINISH_OWNER.lock')
    while not (B/'MANIFEST.json').exists():
        if (B/'BLOCKED.json').exists():raise RuntimeError('recording blocked; see BLOCKED.json')
        time.sleep(5)
    if (B/'SCHEDULING_AMENDMENT01.json').exists():
        deadline=time.monotonic()+1800
        while not (B/'PREFETCH_COMPLETE.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('prefetch completion timeout; do not discard in-flight evidence')
            time.sleep(2)
    cfg=read(B/'MANIFEST.json');sources={};scan_counts={}
    extras=[]
    for label,result in cfg['selection'].items():
        games=[]
        for f in result['scan_files']:
            d=read(f);assert d['status']=='PASS'
            assert all(sha(Path(z))==h for z,h in d['files'].items())
            sources[f]=sha(Path(f));games+=d['games']
        assert [g['source_index'] for g in games]==list(range(result['scanned']))
        want,missing=selected(games)
        assert missing==result['missing']
        assert [(g['index'],g['rule']) for g in want]==[(g['index'],g['rule']) for g in result['selected']]
        assert not missing or result['scanned']==256
        scan_counts[label]=len(games)
        for f in sorted((B/'scan'/label).glob('shard_*/ACCEPTED.json')):
            if str(f) in result['scan_files']:continue
            d=read(f);assert d['status']=='PASS'
            assert all(sha(Path(z))==h for z,h in d['files'].items())
            assert all(result['scanned']<=g['source_index']<256 for g in d['games'])
            extras.append(dict(proof=str(f),proof_sha256=sha(f),games=len(d['games']),reason='bounded prefetch beyond necessary seed prefix; excluded from selection and statistical sample'))
    atomic_json(B/'PREFETCH_ACCEPTANCE.json',dict(status='PASS',selected_prefix_scan_counts=scan_counts,extra_completed_games=sum(x['games'] for x in extras),excluded_shards=extras,protocol_unchanged=True,unix=time.time()))
    run([sys.executable,'-m','scripts.build_new_model_replays_20261031'],'BUILD.log')
    cfg=read(B/'MANIFEST.json');package=read(B/'PACKAGE.json')
    browser_env=dict(os.environ,LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu',BT_REPLAY_BASE=str(B))
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory='.'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        run(['node','scripts/verify_v2_candidate_replays_20261031.js'],'BROWSER.log',dict(browser_env,
            REPLAY_URL=f'http://localhost:{server.server_port}/{B}/{cfg["html"]}',BROWSER_LOCAL='1'))
    finally:server.shutdown();server.server_close()
    local=read(B/'BROWSER.json');assert local['status']=='PASS' and not local['errors']
    uploaded=upload_replay(str(B),cfg['html'])
    assert uploaded['local_sha256']==uploaded['remote_sha256']==package['html_sha256']
    run(['node','scripts/verify_v2_candidate_replays_20261031.js'],'REMOTE_BROWSER.log',
        dict(browser_env,REPLAY_URL=uploaded['url'],BROWSER_LOCAL='0'))
    remote=read(B/'REMOTE_BROWSER.json')
    assert remote['status']=='PASS' and not remote['errors']
    assert len(remote['rows'])==len(cfg['games'])==len(local['rows'])
    assert remote['html_sha256']==uploaded['remote_sha256']
    assert remote['database_writes_blocked'] and remote['all_entries_terminal_seek']
    cmp=read(B/'CMP3040_EXISTING_ACCEPTANCE.json');assert sha(Path(cmp['results']))==cmp['results_sha256']
    acceptance=dict(status='COMPLETED',url=uploaded['url'],html_sha256=uploaded['remote_sha256'],
        entries=len(cfg['games']),independent_games=package['independent_games'],scan_counts=scan_counts,
        missing={k:v['missing'] for k,v in cfg['selection'].items()},models=cfg['models'],
        protocol_sha256=sha(B/'PROTOCOL.json'),manifest_sha256=sha(B/'MANIFEST.json'),
        local_browser_sha256=sha(B/'BROWSER.json'),remote_browser_sha256=sha(B/'REMOTE_BROWSER.json'),
        source_shards=sources,cmp3040=cmp['status'],sp_evidence_sha256=sha(B/'SP_ROLLBACK_EVIDENCE.json'),
        prefetch_acceptance_sha256=sha(B/'PREFETCH_ACCEPTANCE.json'),extra_completed_games=sum(x['games'] for x in extras),
        cpu_only=True,training_workers_signalled=False,web_changed=False,models_uploaded=False,unix=time.time())
    atomic_json(B/'RECEIPT.json',acceptance)
    atomic_json(B/'EXECUTION_STATUS.json',dict(status='COMPLETED',receipt=str(B/'RECEIPT.json'),
        receipt_sha256=sha(B/'RECEIPT.json'),entries=acceptance['entries'],independent_games=acceptance['independent_games'],unix=time.time()))
    report=Path('reports/new_model_replays_20261031.md');text=report.read_text()
    text=text.replace('状态：CPU录制进行中，五模型固定前8局共40局已全部完成原生生产JS Review动作/hash/终局验收。固定40局预览已OSS外网回读SHA一致，40局本地及外网实际播放/终局seek全部通过，console错误0，已京ME通知。负局/平局补集按冻结seed顺序继续；最终交付以主RECEIPT.json为准。',
        f'状态：全部完成。{acceptance["entries"]}条入口/{acceptance["independent_games"]}个独立完整局；本地及OSS外网每入口播放、模型SHA、终局seek通过，console错误0，外网HTTP200逐字节回读一致。固定40局预览另保留。')
    lines=['','## 最终录像验收','','最终自包含HTML：`'+str(B/cfg['html'])+'`；外网Location见 `'+str(B/'RECEIPT.json')+'`，报告仅用 `<REPLAY_URL>`。',
        f'HTML SHA256 `{uploaded["remote_sha256"]}`；{acceptance["entries"]}入口/{acceptance["independent_games"]}独立局。所有选例在原生终局完成，无人工中途截断；记录动作和状态hash重放一致。',
        '', '|模型|按seed扫描完整局数|未找到的补例|','|---|---:|---|']
    for k,v in cfg['selection'].items():lines.append(f'|{cfg["models"][k]["display_name"]}|{v["scanned"]}|{v["missing"]}|')
    lines += ['',f'纯调度预取额外完成{sum(x["games"] for x in extras)}局，均在冻结256局内，原始记录保留但不纳入选例、扫描前缀或统计分母；详见 `runs/new_model_replays_20261031/PREFETCH_ACCEPTANCE.json`。']
    lines+=['','逐局表头、中文时间轴、出生/笼/复活规则与完整hash在页内可见。道具实际释放、接触、自踩链不冒充配对因果净收益；真连锁按原生触发射线事件，非几何连线。','',
        'CPU阶段结束后不控制训练。当前SP是否继续由它自身冻结的bot/资源闸管理，录像任务未更改闸门。没有模型权重上传、commit、push或网页修改。','']
    report.write_text(text+'\n'.join(lines))
    ledger=Path('docs/training_experiment_ledger.md');text=ledger.read_text()
    text=text.replace('补集仍在录制，不能标全部完成。',f'补集已全部完成：{acceptance["entries"]}入口/{acceptance["independent_games"]}独立局，所有补例缺失情况见最终回执；本地和外网逐入口实播/终局seek通过，HTTP回读SHA一致。')
    ledger.write_text(text)
    msg=(f'新模型录像已全部完成：R3-a/b u4002、SP两seed u3316/u3400、网页u3040；'
         f'{acceptance["entries"]}入口/{acceptance["independent_games"]}完整局，固定首8+每席首负/平，JS hard v2本侧。'
         'OSS回读SHA/逐局浏览器播放PASS。cmp3040已2816局完成未重跑，报告reports/new_model_replays_20261031.md。链接：'+uploaded['url'])
    result=subprocess.run(['jdme-notify','send',msg],capture_output=True,text=True)
    atomic_json(B/'NOTIFIED.json',dict(returncode=result.returncode,receipt_sha256=sha(B/'RECEIPT.json'),report_sha256=sha(report),unix=time.time()))
    print(json.dumps(dict(status='COMPLETED',url=uploaded['url'],entries=acceptance['entries'],independent_games=acceptance['independent_games'],notify_returncode=result.returncode),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(B/'DELIVERY_BLOCKED.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e),unix=time.time()))
        raise
