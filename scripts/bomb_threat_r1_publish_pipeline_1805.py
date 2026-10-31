"""Persist the authorized R1 selection -> replay -> verified release successor."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json, lock, sha
from scripts.bomb_threat_round15 import read as _read

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('runs/bomb_threat_v1')
OUT = BASE / 'r1_fresh_publish_1805_20261030'
WORKTREE = Path('../cx_publish_r1_1805_20261030')


def read(path, default=None):
    return _read(Path(path), default)


def module_running(module):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if module.encode() in (proc/'cmdline').read_bytes().split(b'\0') and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':
                return True
        except OSError:
            pass
    return False


def launch_cpu(module, logname):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu')
    with (OUT/logname).open('ab',buffering=0) as log:
        subprocess.Popen(['taskset','-c','128-131',sys.executable,'-m',module],cwd=ROOT,
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def ensure_running():
    if not (OUT/'PROTOCOL.json').exists() or (OUT/'DONE.json').exists() or (OUT/'PIPELINE_BLOCKED.json').exists():
        return
    module='scripts.bomb_threat_r1_publish_pipeline_1805'
    if not module_running(module):
        launch_cpu(module,'PIPELINE.log')


def prepare():
    from scripts import bomb_threat_a3sr_lineage_watch as L
    from scripts.bomb_threat_r1_fresh_publish_1805 import collect
    decision = read(OUT / 'FINAL_SELECTION.json')
    assert decision['publish']
    source = decision['models'][decision['winner']]
    slot, update = source['slot'], source['update']
    ident = f'bun392-r1-slot{slot:02d}-u{update}'
    tag = f'web-{ident}-20261030'
    model = read(source['model_json'])
    graph = Path(source['model_json']).with_name('model.onnx')
    assert sha(source['model_json']) == source['model_sha256']
    assert model['meta']['checkpoint_sha256'] == source['policy_sha256']
    assert sha(graph) == model['meta']['onnx_sha256']
    model['meta'].pop('onnx_path', None)
    model['meta'].update(name=f'Bun392 R1 slot{slot:02d} u{update}',
        display_name=f'Bun392 R1 slot{slot:02d} u{update}', source=source['checkpoint']+'/policy.pt',
        committed_sha256=sha(Path(source['checkpoint'])/'COMMITTED.json'),
        strategy='raw', state_dim=24, attention_heads=4, update=update, rule_version='stock-window-v1')
    data=(json.dumps(model,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    h=hashlib.sha256(data).hexdigest();gh=sha(graph)
    filename=f'{ident}-{h}.json';gname=f'{ident}-{gh}.onnx'
    (OUT/filename).write_bytes(data);(OUT/gname).write_bytes(graph.read_bytes())
    scores=decision['comparison'];selected=scores[decision['winner']];baseline=scores['baseline']
    url=f'https://github.com/allwellll/qqt-RL/releases/download/{tag}/'
    desc='；'.join(f'{tier} P1 {selected["P1"][tier]:+d}（网页原模型{baseline["P1"][tier]:+d}）' for tier in ('hard','normal','easy'))
    row=dict(id=ident,display_name=f'Bun392 R1 slot{slot:02d} u{update}（P1三档{selected["P1_sum"]:+d}）',
        file=filename,sha256=h,bytes=len(data),url=url+filename,cycle=update,
        candidate=Path(source['checkpoint']).parent.name+f'_u{update:06d}_raw',source_sha256=source['policy_sha256'],
        strategy='raw',score=decision['results'][decision['winner']]['hard']['win']/256,
        description='网页真人P0、模型P1；生产JS Hunter、stock-window-v1、argmax、老师关闭。'+desc+
          f'；P1合计{selected["P1_sum"]:+d} vs原模型{baseline["P1_sum"]:+d}；P0合计{selected["P0_sum"]:+d} vs {baseline["P0_sum"]:+d}。逐档/P0/自伤护栏通过；固定候选三档各256局、每席128。64几何复用，不宣称统计显著或真人泛化。R1原生道具训练规则与JS部署能力分开；训练bf16与部署fp32存在差异；仍可能有零泡、振荡、持包停滞。',
        score_protocol='production JS bun.hunter; stock-window-v1; argmax; teacher off; paired256 per tier P0/P1 128 each; hard2026193000 normal2026194000 easy2026195000; hard wins/256;64 geometries reused',
        onnx=dict(file=gname,sha256=gh,bytes=graph.stat().st_size,url=url+gname,meta=model['meta']))
    atomic_json(OUT/'CATALOG_ROW.json',row)
    atomic_json(OUT/'manifest.json',dict(schema='qqt.bun-release/v1',tag=tag,source_ref=source['checkpoint'],
        policy_sha256=source['policy_sha256'],committed_sha256=sha(Path(source['checkpoint'])/'COMMITTED.json'),
        rule_version='stock-window-v1',strategy='raw',selection_sha256=sha(OUT/'FINAL_SELECTION.json'),
        raw_tensors_unchanged=model['flat']==read(source['model_json'])['flat'] and model['tensors']==read(source['model_json'])['tensors'],
        comparison=scores,assets=[dict(file=filename,sha256=h,bytes=len(data)),dict(file=gname,sha256=gh,bytes=graph.stat().st_size)]))
    (OUT/'SHA256SUMS').write_text(f'{h}  {filename}\n{gh}  {gname}\n{sha(OUT/"manifest.json")}  manifest.json\n')
    # Replay selection uses the SAME hard seed/birth list as the current page.
    old=read(BASE/'iter2/a3sr/lineage_watch_20261030/PROTOCOL.json')
    tier=old['fresh']['tiers']['hard'];root=ROOT/OUT
    L.OUT=root
    jobs=[dict(kind='replay_selection',slot=slot,update=update,tier='hard',start=i,
        games=256,seed=tier['seed'],spawn_manifest=tier['spawn_manifest']) for i in range(0,256,8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda j:collect(j,source),jobs))
    rows=[];primary={};files={}
    for job in jobs:
        p=L.shard_path(job);proof=read(str(p)+'.ACCEPTED.json');files.update(proof['files'])
        rr,eps=L.validate_shard(p,job,source);rows+=rr;primary.update({e['seed']:e['delivery'] for e in eps})
    rows.sort(key=lambda r:r['seed']);games=[];missing=[]
    def add(r,rule):
        item=copy.deepcopy(r);item['primary_delivery']=primary[r['seed']]
        games.append(dict(index=r['seed']-tier['seed'],seed=r['seed'],seat=r['seat'],rule=rule,expected=item))
    for r in rows[:8]:add(r,'固定首8局，与当前网页录像同seed')
    for seat in (0,1):
        wins=[r for r in rows if r['seat']==seat and r['winner']==seat][:3]
        losses=[r for r in rows if r['seat']==seat and r['winner'] not in (None,-1,seat)][:1]
        for r in wins:add(r,f'P{seat}最早3胜')
        for r in losses:add(r,f'P{seat}最早1负')
        if len(wins)<3 or not losses:missing.append(dict(seat=seat,wins=len(wins),losses=len(losses)))
    zeros=[r for r in rows if r['seat']==1 and r['placed'][1]==0 and r['placed'][0]>0][:2]
    for r in zeros:add(r,'最早2个P1单方零泡')
    if len(zeros)<2:missing.append(dict(P1_single_zero=len(zeros)))
    for prior in read(BASE/'iter2/a3sr/BRICK_OBSERVATION_MANIFEST.json')['games'][:3]:
        add(next(r for r in rows if (r['seed'],r['seat'])==(prior['seed'],prior['seat'])),'旧u800隔砖清单前3局同seed')
    replay=BASE/f'replay_r1_slot{slot:02d}_u{update}_1805_20261030'
    atomic_json(replay/'MANIFEST.json',dict(status='FROZEN_BEFORE_RECORDING',model=source,games=games,missing=missing,
        source_files=files,spawn_manifest=tier['spawn_manifest'],model_label=f'R1 slot{slot:02d} u{update} · 网页默认候选',
        html=f'20261030-r1-slot{slot:02d}-u{update}-review.html',teacher=False,rule_version='stock-window-v1',unix=time.time()))
    atomic_json(OUT/'REPLAY_PREPARED.json',dict(base=str(replay),manifest_sha256=sha(replay/'MANIFEST.json'),html=read(replay/'MANIFEST.json')['html']))


def report(row, receipt, head, actions, replay, browser):
    selection=read(OUT/'FINAL_SELECTION.json');p=Path('reports')/f'publish_best_{row["id"]}_20261030.md'
    lines=['# R1固定候选复核与发布 — 20261030','',
        '状态：Release/tag、非force main快进、同head Pages、线上完整资产SHA与实际P1整局、新Review录像浏览器和OSS回读核验通过。健康训练继续。','',
        '当前网页旧模型为A3SR slot00 c34 u2400；固定候选slot07 c35 u2960、slot00 c35 u2920、slot03 c36 u2760。相同生产JS Hunter/stock-window-v1/argmax/老师关闭，hard/normal/easy每模型256局，P0/P1各128。候选新增2304局，基线768局同seed原始分片SHA验收复用；不择峰、不混新旧seed。64几何复用，不宣称显著性或真人泛化。','',
        '| 模型 | P1 hard/normal/easy | P1合计 | P0合计 | 自伤 | 护栏 |','|---|---|---:|---:|---:|---|']
    for label,x in selection['comparison'].items():
        lines.append(f'|{label}|{x["P1"]}|{x["P1_sum"]}|{x["P0_sum"]}|{x["self_deaths"]}|{x["guards"]}|')
    lines += ['',f'发布：`{row["id"]}`，policy `{row["source_sha256"]}`，JSON `{row["sha256"]}`，ONNX `{row["onnx"]["sha256"]}`。',
        f'部署SHA `{head}`；tag `{receipt["tag"]}`；Release `{receipt["url"]}`；Pages `{actions["html_url"]}`。',
        '线上 https://allwellll.github.io/qqt-RL/ 。旧模型、旧录像和规则保留；仅模型目录与必要测试断言进入提交，隐私扫描通过。',
        f'录像原始核验与URL见 `{replay["base"]}/RECEIPT.json`；逐局连旧泡、异时序连接、有效威胁、近敌3格、终局三属性见 `R1_REPLAY_METRICS.json`，表头公开定义、分母及未知；不把观测相关当作因果收益或训练奖励。',
        '浏览器数据库写入屏蔽，线上P1完整局动作finite；已验SHA的模型缓存与冷网性能范围分开披露，性能原始数值见ONLINE_BROWSER_CACHED.json。',
        f'完整选择/导出/回归/浏览器/隐私/同head部署/线上HASH证据 `{OUT}`。','']
    p.write_text('\n'.join(lines));return p


def retain_baseline_replay(selection):
    import functools
    import http.server
    import shutil
    import threading
    from scripts.best_publish_release_20261030 import upload_replay
    original=read(BASE/'publish_best_20261030/REPLAY_PREPARED.json')
    source=Path(original['base']);dest=BASE/'replay_r1_comparison_baseline_1805_20261030'
    dest.mkdir(parents=True,exist_ok=True)
    cfg=read(source/'MANIFEST.json');cfg.update(html='20261030-r1-comparison-current-baseline-review.html',
        model_label='当前网页A3SR slot00 u2400 · R1复核保留基线',
        comparison_selection=str(OUT/'FINAL_SELECTION.json'))
    for index in {g['index'] for g in cfg['games']}:
        src=source/f'game_{index:03d}';target=dest/f'game_{index:03d}';target.mkdir(exist_ok=True)
        for name in ('review.json','MATCH.json'):
            shutil.copy2(src/name,target/name)
    atomic_json(dest/'MANIFEST.json',cfg)
    env=dict(os.environ,BT_REPLAY_BASE=str(dest),CUDA_VISIBLE_DEVICES='')
    def run(args,log,extra=None):
        with (OUT/log).open('ab') as f:
            subprocess.run(args,env=dict(env,**(extra or {})),stdout=f,stderr=subprocess.STDOUT,check=True)
    run(['node','scripts/bomb_threat_r1_replay_metrics.js',str(dest)],'BASELINE_REPLAY_METRICS.log')
    run([sys.executable,'-m','scripts.build_bomb_threat_a3sr_replay'],'BASELINE_REPLAY_BUILD.log')
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(ROOT))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        run(['node','scripts/verify_bomb_threat_a3sr_replay_browser.js'],'BASELINE_REPLAY_BROWSER.log',
            dict(BROWSER_LOCAL='1',REPLAY_URL=f'http://127.0.0.1:{server.server_port}/'+str(dest)+'/'+cfg['html']))
    finally:
        server.shutdown();server.server_close()
    uploaded=upload_replay(str(dest),cfg['html'])
    run(['node','scripts/verify_bomb_threat_a3sr_replay_browser.js'],'BASELINE_REPLAY_REMOTE_BROWSER.log',
        dict(REPLAY_URL=uploaded['verification_url']))
    atomic_json(dest/'RECEIPT.json',dict(status='COMPLETED',url=uploaded['url'],
        html_sha256=uploaded['remote_sha256'],reused_verified_recordings=True,
        metrics_recomputed=True,old_page_preserved=True,unix=time.time()))
    p=Path('reports/r1_fresh_comparison_20261030.md')
    lines=['# R1三候选固定复核 — 20261030','','三候选均未满足超过当前网页模型及全部安全护栏，保留默认模型。固定同seed生产JS三档各256局，P0/P1各128；新增2304局，原网页同协议768局SHA验收复用。64几何复用，不宣称显著性。','',
        '| 模型 | P1三档 | P1合计 | P0合计 | 自伤 | 护栏 |','|---|---|---:|---:|---:|---|']
    for label,x in selection['comparison'].items():
        lines.append(f'|{label}|{x["P1"]}|{x["P1_sum"]}|{x["P0_sum"]}|{x["self_deaths"]}|{x["guards"]}|')
    lines+=['',f'完整比较 `{OUT}/FINAL_SELECTION.json`。新录像使用当前网页同一批已完整核验的Review，旧页不变；添加R1表头/分母/定义，由Review动作重算并验证终态hash，不冒称新采集或训练奖励。URL及HTML/hash/浏览器证据 `{dest}/RECEIPT.json`。健康训练继续。','']
    p.write_text('\n'.join(lines))
    return uploaded['url']


def main():
    guard=lock(OUT/'PIPELINE.lock')
    while not (OUT/'EVALUATION_COMPLETE.json').exists():
        if read(OUT/'BLOCKED.json',{}).get('status')=='BLOCKED':
            raise RuntimeError('evaluation blocked; inspect relative BLOCKED.json')
        module='scripts.bomb_threat_r1_fresh_publish_1805'
        if not module_running(module):
            launch_cpu(module,'OWNER.log')
        atomic_json(OUT/'PIPELINE_STATUS.json',dict(status='WAITING_FIXED_SELECTION',training_untouched=True,unix=time.time()))
        time.sleep(15)
    result=read(OUT/'FINAL_SELECTION.json')
    assert read(OUT/'EVALUATION_COMPLETE.json')['selection_sha256']==sha(OUT/'FINAL_SELECTION.json')
    if not result['publish']:
        replay_url=retain_baseline_replay(result)
        atomic_json(OUT/'DONE.json',dict(status='COMPLETED_RETAINED_BASELINE',selection_sha256=sha(OUT/'FINAL_SELECTION.json'),replay_url=replay_url,unix=time.time()))
        subprocess.run(['jdme-notify','send','R1三候选完整配对复核完成，无候选满足全部提升/安全护栏，保留当前网页模型；新指标录像 '+replay_url+'；训练继续。'],check=True)
        return
    # Publishing must still target the exact verified original online head.
    from scripts.a3sr_publish_github import request
    remote=request('git/ref/heads/main')['object']['sha']
    previous=read(BASE/'publish_best_20261030/DONE.json')['head']
    assert remote==previous,'main advanced: preserve and revalidate instead of overwriting'
    if not WORKTREE.exists():
        subprocess.run(['git','worktree','add','--detach',str(WORKTREE),remote],check=True,stdout=subprocess.DEVNULL)
    assert subprocess.check_output(['git','-C',str(WORKTREE),'rev-parse','HEAD']).decode().strip()==remote
    # A retry may already have staged the selected row in this worktree.
    # The immutable baseline is the verified online HEAD, not staged contents.
    original_catalog=json.loads(subprocess.check_output(
        ['git','-C',str(WORKTREE),'show',remote+':web/models.json']))
    staged_catalog=read(WORKTREE/'web/models.json')
    assert (staged_catalog==original_catalog or
            staged_catalog['models'][1:]==original_catalog['models']), 'unexpected staged catalog edits'
    atomic_json(OUT/'ONLINE_CATALOG_BEFORE.json',original_catalog)
    atomic_json(OUT/'FINAL_SELECTION_COMPLETE.json',dict(status='COMPLETE',selection_sha256=sha(OUT/'FINAL_SELECTION.json')))
    os.environ.update(BT_PUBLICATION_BASE=str(OUT),BT_PUBLICATION_WORKTREE=str(WORKTREE),
        BT_PUBLICATION_PREPARE_MODULE='scripts.bomb_threat_r1_publish_pipeline_1805',
        BT_PUBLICATION_PREPARE_FLAG='--prepare',BT_REPLAY_R1_METRICS='1')
    from scripts import best_publish_prepublication_20261030 as pre
    pre.O=OUT;pre.W=WORKTREE;pre.main()
    from scripts import best_publish_release_20261030 as release
    release.O=OUT;release.W=WORKTREE;release.report=report
    release.main()


if __name__=='__main__':
    try:
        prepare() if '--prepare' in sys.argv else main()
    except Exception as error:
        atomic_json(OUT/'PIPELINE_BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,
            reason='inspect PIPELINE.log; training and old online default preserved unless prior release receipts show otherwise',unix=time.time()))
        subprocess.run(['jdme-notify','send','R1复核/发布后继出现技术阻塞：'+type(error).__name__+'；已保存PIPELINE_BLOCKED，健康训练不受影响。'],capture_output=True)
        raise
