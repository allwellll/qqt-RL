"""Authorized v2 release, gated by complete paired results and remote rule boards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import functools
import http.server
import json
import hashlib
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
from scripts.a3sr_publish_github import request
from scripts.best_publish_release_20261030 import upload_replay
B=Path(os.environ.get('V2_SELECTION_OUTPUT','runs/v2_fresh_selection_20261031'));O=B/'publication';W=Path('../cx_publish_r1_1805_20261030')
SIM='f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc'
def read(p):return json.loads(Path(p).read_text())
def run(args,log,env=None,cwd=None):
 with (O/log).open('ab') as stream:subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,**(env or {})),cwd=cwd,check=True)
def git(*args):return subprocess.check_output(['git','-C',str(W),*args]).decode().strip()
def status(value,**kw):atomic_json(O/'STATUS.json',dict(status=value,unix=time.time(),**kw))
def notify(message):subprocess.run(['jdme-notify','send',message],capture_output=True,check=False)
def remote_gate():
 cfg=(W/'web/leaderboard_config.js').read_text();url=re.search("url: '([^']+)'",cfg)[1];key=re.search("publishableKey: '([^']+)'",cfg)[1]
 def rpc(name,payload):
  req=urllib.request.Request(url+'/rest/v1/rpc/'+name,data=json.dumps(payload).encode(),headers={'apikey':key,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=45) as r:return json.load(r)
 try:
  contract=rpc('qqt_ruleset_contract',{});assert contract['ruleset_contract_version']==2
  boards={}
  for rule in ('stock-window-v1','bun-native-v2'):
   x=rpc('qqt_my_win_leaderboard_ruleset',dict(p_player_id=str(uuid.uuid4()),p_player_secret='0'*64,p_ruleset_version=rule))
   assert x['ruleset_contract_version']==2 and x['ruleset_version']==rule
   boards[rule]=dict(rows=len(x['rows']),contract=2)
  result=dict(status='PASS',contract=contract,boards=boards,read_only=True,unix=time.time())
 except (urllib.error.HTTPError,AssertionError,KeyError) as error:
  result=dict(status='BLOCKED_REMOTE_RULESET_MIGRATION',error_type=type(error).__name__,http_status=getattr(error,'code',None),migration='supabase/migrations/20261031074000_ruleset_boards.sql',read_only=True,unix=time.time())
 atomic_json(O/'REMOTE_DATABASE_GATE.json',result);return result['status']=='PASS'
def browser_replay(base,html):
 from scripts.check_publish_privacy import findings
 parts=Path.cwd().parts;tokens=[parts[parts.index('afs')+1]] if 'afs' in parts else []
 raw=(base/html).read_text()
 assert not findings(raw,tokens),'public replay privacy audit failed'
 assert not re.search(r'/mnt/jpfs/afs/|/home/[^/]+/',raw),'public replay contains local path'
 env=dict(BT_REPLAY_BASE=str(base),LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu')
 handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory='.')
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler);threading.Thread(target=server.serve_forever,daemon=True).start()
 try:run(['node','scripts/verify_v2_candidate_replays_20261031.js'],'REPLAY_LOCAL_BROWSER.log',dict(env,BROWSER_LOCAL='1',REPLAY_URL=f'http://127.0.0.1:{server.server_port}/{base}/{html}'))
 finally:server.shutdown();server.server_close()
 upload=upload_replay(base,html)
 run(['node','scripts/verify_v2_candidate_replays_20261031.js'],'REPLAY_REMOTE_BROWSER.log',dict(env,REPLAY_URL=upload['url']))
 atomic_json(base/'RECEIPT.json',dict(status='COMPLETED',url=upload['url'],html_sha256=upload['remote_sha256'],browser_sha256=sha(base/'REMOTE_BROWSER.json'),rule_version='bun-native-v2',unix=time.time()))
 return upload

def prepare():
 status('PREPARING_EXACT_WINNER_ASSETS')
 run([sys.executable,'-m','scripts.v2_release_prepare_20261031'],'PREPARE.log')
 if read(O/'STATUS.json')['status']=='RETAIN_BASELINE_NO_ELIGIBLE_CANDIDATE':return False
 run(['node','scripts/verify_v2_release_export_20261031.js'],'JS_ONNX.log')
 run([sys.executable,'-m','scripts.v2_release_fp32_audit_20261031'],'PYTHON_PARITY.log')
 row=read(O/'CATALOG_ROW.json');baseline=read(B/'ONLINE_BASELINE_VERIFIED.json')
 base=baseline['build']['commit'];assert git('rev-parse','HEAD')==base or (O/'LOCAL_COMMIT.json').exists()
 old=json.loads(subprocess.check_output(['git','-C',str(W),'show',base+':web/models.json']))
 atomic_json(O/'ONLINE_CATALOG_BEFORE.json',old)
 assert old['models'][0]['sha256']==baseline['model_sha256']
 staged=read(W/'web/models.json');assert staged==old or (staged['models'][0]['id']==row['id'] and staged['models'][1:]==old['models'])
 catalog=dict(old,models=[row,*old['models']]);atomic_json(W/'web/models.json',catalog)
 for asset in (row,row['onnx']):
  dest=W/'web/models'/asset['file'];dest.parent.mkdir(exist_ok=True)
  if not dest.exists():dest.symlink_to((O/asset['file']).resolve())
  assert sha(dest)==asset['sha256']
 p=W/'web/test_model_catalog.js';s=p.read_text()
 for field,value in [('id',row['id']),('source_sha256',row['source_sha256']),('score',row['score'])]:s=re.sub(r'assert\.strictEqual\(current\.'+field+r',[^;]+;',f'assert.strictEqual(current.{field},{json.dumps(value)});',s)
 s=s.replace("assert.strictEqual(current.onnx.meta.rule_version,'stock-window-v1');","assert.strictEqual(current.onnx.meta.rule_version,'bun-native-v2');")
 p.write_text(s)
 p=W/'web/test_http.js';s=p.read_text();s=re.sub(r'assert\.strictEqual\(manifest\.models\.length,\s*\d+\);',f'assert.strictEqual(manifest.models.length, {len(catalog["models"])});',s);s=re.sub(r'assert\.strictEqual\(manifest\.models\[0\]\.id,\s*[^;]+;',f'assert.strictEqual(manifest.models[0].id, {json.dumps(row["id"])});',s);p.write_text(s)
 run([sys.executable,'-m','scripts.v2_final_replay_prepare_20261031'],'REPLAY_PREPARE.log')
 replay=B/'final_replay';cfg=read(replay/'MANIFEST.json')
 def record(index):
  folder=replay/f'game_{index:03d}';folder.mkdir(exist_ok=True)
  if (folder/'MATCH.json').exists():
   receipt=read(folder/'MATCH.json');assert receipt['status']=='MATCH' and receipt['review_sha256']==sha(folder/'review.json');return
  run(['node','scripts/v2_final_replay_record_20261031.js',str(index)],f'RECORD_{index:03d}.log',dict(BT_REPLAY_BASE=str(replay)))
  assert read(folder/'MATCH.json')['status']=='MATCH'
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(record,sorted({x['index'] for x in cfg['games']})))
 atomic_json(replay/'RECORDING_COMPLETE.json',dict(status='MATCH',unique_games=len({x['index'] for x in cfg['games']}),manifest_sha256=sha(replay/'MANIFEST.json'),unix=time.time()))
 run(['node','scripts/v2_replay_metrics_20261031.js',str(replay)],'REPLAY_METRICS.log')
 run(['node','scripts/v2_candidate_replay_analysis_20261031.js'],'REPLAY_OBSERVATIONS.log',dict(BT_REPLAY_BASE=str(replay)))
 run([sys.executable,'-m','scripts.build_v2_candidate_replays_20261031'],'REPLAY_BUILD.log',dict(V2_REPLAY_BASE=str(replay)))
 upload=browser_replay(replay,cfg['html'])
 atomic_json(O/'REPLAY.json',dict(base=str(replay),html=cfg['html'],url=upload['url']))
 notify('v2固定矩阵胜出候选完整录像已验收：'+row['id']+'；'+upload['url']+'。包含固定首8局及披露选择的胜/负/零泡局；尚未宣布线上默认切换，训练不停。')
 return True

def preflight():
 if not remote_gate():
  status('BLOCKED_REMOTE_RULESET_MIGRATION',resume_command='<PYTHON> -m scripts.v2_web_release_20261031',training_untouched=True)
  notify('v2固定矩阵与候选录像准备完成；默认切换阻塞：远端Supabase ruleset分榜迁移尚未通过。需执行既已提供增量SQL；旧网页/数据保留，训练不停。');return False
 status('FINAL_LOCAL_REGRESSION_AND_BROWSER')
 run(['npm','test'],'NPM_TEST.log',cwd=W)
 assert sha(W/'web/sim-bun-native-v2.js')==SIM
 handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(W/'web'))
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler);threading.Thread(target=server.serve_forever,daemon=True).start()
 try:run(['node','scripts/verify_v2_publication_browser_20261031.js'],'LOCAL_BROWSER.log',dict(BT_PUBLICATION_BASE=str(O),LOCAL='1',WEB_URL=f'http://127.0.0.1:{server.server_port}/',LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu'))
 finally:server.shutdown();server.server_close()
 run(['git','-C',str(W),'diff','--check'],'DIFF_CHECK.log')
 changed=git('diff','--name-only').splitlines()+git('ls-files','--others','--exclude-standard').splitlines()
 allowed=read(B/'PUBLICATION_PROTOCOL.json')['allowed_files'];assert set(changed)<=set(allowed),(set(changed)-set(allowed))
 # Only git-listed files plus explicit release assets, never scan local run trees.
 from scripts.check_publish_privacy import scan
 parts=Path.cwd().parts;tokens=[parts[parts.index('afs')+1]] if 'afs' in parts else []
 findings=scan(W,tokens,include=[x for x in changed if x not in git('ls-files').splitlines()])
 for asset in ('CATALOG_ROW.json','manifest.json',read(O/'CATALOG_ROW.json')['file']):
  raw=(O/asset).read_text();assert not re.search(r'/mnt/jpfs/afs/|/home/[^/]+/|10\.186\.\d+\.\d+',raw)
 assert not findings,findings
 atomic_json(O/'PREPUBLICATION_ACCEPTANCE.json',dict(status='PASS',packed_tensor_exact=True,JS_ONNX='PASS',Python_fp32='PASS',Chromium='PASS',npm_test='PASS',database_ruleset_contract=2,privacy_findings=findings,changed_files=changed,training_untouched=True,unix=time.time()))
 return True

def release():
 row=read(O/'CATALOG_ROW.json');manifest=read(O/'manifest.json');tag=manifest['tag'];acceptance=read(O/'PREPUBLICATION_ACCEPTANCE.json');assert acceptance['status']=='PASS'
 assert remote_gate(),'remote ruleset database gate regressed'
 if not (O/'LOCAL_COMMIT.json').exists():
  base=git('rev-parse','HEAD');assert request('git/ref/heads/main')['object']['sha']==base,'remote main advanced; preserve and revalidate'
  files=acceptance['changed_files'];run(['git','-C',str(W),'add','--',*files],'ADD.log')
  run(['git','-C',str(W),'-c','user.name=QQT Model Release','-c','user.email=model-release@users.noreply.github.com','commit','-m','Publish paired-validated bun-native-v2 default with separate ruleset boards'],'COMMIT.log')
  atomic_json(O/'LOCAL_COMMIT.json',dict(status='COMMITTED',base=base,head=git('rev-parse','HEAD'),changed_files=files,unix=time.time()))
 head=read(O/'LOCAL_COMMIT.json')['head'];assert git('rev-parse','HEAD')==head
 if not git('tag','--list',tag):run(['git','-C',str(W),'-c','user.name=QQT Model Release','-c','user.email=model-release@users.noreply.github.com','tag','-a',tag,head,'-m','bun-native-v2; paired production JS; argmax; old v1 and results retained'],'TAG.log')
 assert git('rev-list','-n','1',tag)==head
 env=dict(BT_PUBLICATION_BASE=str(O),BT_PUBLICATION_WORKTREE=str(W),BT_RELEASE_TAG=tag)
 if not (O/'GIT_OBJECT_IMPORT.json').exists():run([sys.executable,'-m','scripts.a3sr_git_transport','import'],'GIT_IMPORT.log',env)
 body=row['description']+f"\n\nPolicy SHA256: `{row['source_sha256']}`\nJSON SHA256: `{row['sha256']}`\nONNX SHA256: `{row['onnx']['sha256']}`\n\nRule SHA256: `{SIM}`\nOld stock-window-v1 remains selectable and its results remain separate.\n"
 (O/'RELEASE.md').write_text(body)
 try:r=request('releases/tags/'+tag)
 except urllib.error.HTTPError as error:
  if error.code!=404:raise
  r=request('releases','POST',json.dumps(dict(tag_name=tag,target_commitish=head,name=row['display_name']+' · bun-native-v2',body=body,draft=False,prerelease=False)).encode())
 assets={x['name']:x for x in r['assets']}
 for name in [a['file'] for a in manifest['assets']]+['manifest.json','SHA256SUMS']:
  if name not in assets:assets[name]=request(r['upload_url'].split('{')[0]+'?name='+name,'POST',(O/name).read_bytes(),'application/octet-stream')
  assert assets[name]['size']==(O/name).stat().st_size
  if assets[name].get('digest'):assert assets[name]['digest']=='sha256:'+sha(O/name)
 atomic_json(O/'RELEASE_RECEIPT.json',dict(status='RELEASE_ASSETS_UPLOADED',tag=tag,url=r['html_url'],head=head,assets=[{k:a.get(k) for k in ('name','size','digest')} for a in assets.values()],unix=time.time()))
 if not (O/'MAIN_PUSH.json').exists():run([sys.executable,'-m','scripts.a3sr_git_transport','main'],'MAIN_PUSH.log',env)
 status('WAITING_SAME_HEAD_PAGES',head=head)
 end=time.monotonic()+3600
 while time.monotonic()<end:
  rows=request('actions/runs?head_sha='+head+'&per_page=20')['workflow_runs'];pages=[x for x in rows if x['name']=='Deploy Bun rule-bot web to Pages']
  atomic_json(O/'ACTIONS.json',dict(head=head,runs=[{k:x[k] for k in ('name','head_sha','status','conclusion','html_url')} for x in rows],unix=time.time()))
  if pages and pages[0]['status']=='completed':assert pages[0]['conclusion']=='success';break
  time.sleep(20)
 else:raise RuntimeError('same-head Pages deadline exceeded')
 page='https://allwellll.github.io/qqt-RL/'
 def get(name):
  with urllib.request.urlopen(page+name+'?v2verify='+str(time.time_ns()),timeout=180) as response:return response.read()
 build=json.loads(get('build-info.json'));assert build['commit']==head
 catalog=json.loads(get('models.json'));assert catalog['models'][0]==row
 assert hashlib.sha256(get('sim-bun-native-v2.js')).hexdigest()==SIM
 for a in (row,row['onnx']):assert hashlib.sha256(get('models/'+a['file'])).hexdigest()==a['sha256']
 atomic_json(O/'ONLINE_ASSETS.json',dict(status='PASS',head=head,model_sha256=row['sha256'],onnx_sha256=row['onnx']['sha256'],rule_version='bun-native-v2',simulator_sha256=SIM,unix=time.time()))
 run(['node','scripts/verify_v2_publication_browser_20261031.js'],'ONLINE_BROWSER.log',dict(env,WEB_URL=page,EXPECTED_HEAD=head,LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu'))
 replay=read(O/'REPLAY.json');done=dict(status='PUBLISHED_VERIFIED',head=head,tag=tag,policy_sha256=row['source_sha256'],model_sha256=row['sha256'],onnx_sha256=row['onnx']['sha256'],ruleset='bun-native-v2',webpage_url=page,replay_url=replay['url'],release_url=r['html_url'],database_ruleset_contract=2,training_untouched=True,unix=time.time())
 atomic_json(O/'DONE.json',done);status('PUBLISHED_VERIFIED',head=head)
 report=Path('reports/publish_best_bun_native_v2_20261031.md');report.write_text('# bun-native-v2 条件发布验收（20261031）\n\n'+row['description']+'\n\n'+json.dumps(done,ensure_ascii=False,indent=2)+'\n\n完整6144局比较见本轮本侧出生复核报告；原始SHA、完整state来源、导出逐张量、fp32/JS/ONNX/浏览器、回归、分榜、非force Git、同head Pages和冷加载完整P1真人模式证据见本轮独立选型namespace的publication目录。数据库写入在浏览器测试中屏蔽。真人键盘探针1秒后空手等待，不能冒充真人对抗能力评估。旧模型/规则/回放/数据保留；未修改训练协议或停止训练。\n')
 with Path('docs/training_experiment_ledger.md').open('a') as f:f.write('\nV2发布终态：'+row['id']+'，policy `'+row['source_sha256']+'`，commit `'+head+'`，bun-native-v2、分榜contract2、冷加载及完整P1真人模式核验PASS；见 `reports/publish_best_bun_native_v2_20261031.md`。\n')
 notify('v2最好模型与规则同步发布核验完成：'+row['id']+'；网页 '+page+'；新录像 '+replay['url']+'；commit '+head+'。训练不停。')

def main():
 O.mkdir(exist_ok=True);guard=lock(O/'OWNER.lock')
 if (O/'USER_FREEZE.json').exists() or (B/'INVALIDATED.json').exists():
  status('FROZEN_INVALIDATED_SPAWN_PROTOCOL',training_untouched=True)
  return
 protocol=read(B/'PUBLICATION_PROTOCOL.json')
 assert all(sha(Path(f))==h for f,h in protocol['source_sha256'].items()),'publication source changed since frozen protocol'
 assert all(sha(W/f)==h for f,h in protocol['prepared_web_sha256'].items()),'release worktree source changed; preserve edits and review'
 if (O/'DONE.json').exists():return
 if not (O/'REPLAY.json').exists():
  if not prepare():
   atomic_json(O/'DONE.json',dict(status='RETAINED_BASELINE_NO_ELIGIBLE_CANDIDATE',results_sha256=sha(B/'RESULTS.json'),unix=time.time()));notify('v2五候选固定复核均未满足发布标准，保留旧网页模型与规则；报告已写，候选预览仍可播放，训练不停。');return
 if not preflight():return
 release()
if __name__=='__main__':
 try:main()
 except Exception as error:
  O.mkdir(exist_ok=True);atomic_json(O/'BLOCKED.json',dict(status='BLOCKED',error_type=type(error).__name__,reason='inspect OWNER.log; no forced ref or training signal issued',unix=time.time()));notify('v2条件发布链技术阻塞：'+type(error).__name__+'；已写publication/BLOCKED.json与日志，保留现场和健康训练。');raise
