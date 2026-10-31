"""Package, double-read and browser-verify only this authorized preview."""
import functools,hashlib,http.server,json,os,shutil,subprocess,sys,threading,time,urllib.request
from pathlib import Path
from scripts.population_run_state import atomic_json,lock,sha
B=Path('runs/bomb_threat_v1/replay_r1_slot00_u2400_preview')
def read(p):return json.loads(Path(p).read_text())
def run(args,env,log):
 with (B/log).open('ab') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def summary(g,root):
 m=read(root/f"game_{g['index']:03d}/PASSIVE_METRICS.json");r=read(root/f"game_{g['index']:03d}/MATCH.json")['expected'];p=g['seat'];w=r['winner']
 return dict(放泡=r['placed'][p],连上旧泡='未记录',异时序批次威胁='未记录',有效威胁='未记录',平均绕墙距离=round(m['mean_wall_path_distance'],2) if m['mean_wall_path_distance'] is not None else '不可达/未记录',三格内比例=round(m['within3_fraction'],3) if m['within3_fraction'] is not None else '不可达/未记录',距离有效帧=m['distance_reachable_ticks'],双活帧=m['both_alive_ticks'],不可达帧=m['unreachable_ticks'],属性终值=m['terminal_stats'],因果击杀=r['causal_kills'][p],交付=r['primary_delivery'],自伤=r['self_deaths'][p],胜负='平' if w in (None,-1) else '胜' if w==p else '负')
def main():
 guard=lock(B/'FINISH_OWNER.lock');env=dict(os.environ,BT_REPLAY_BASE=str(B),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/tmp/qqt-browser-libs/root/usr/lib/x86_64-linux-gnu'+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
 while not (B/'RECORDING_COMPLETE.json').exists():
  if (B/'FAILURE.json').exists():raise RuntimeError('Recorder failed; see FAILURE.json')
  time.sleep(5)
 assert read(B/'RECORDING_COMPLETE.json')['status']=='MATCH'
 cfg=read(B/'MANIFEST.json');rows=[]
 for g in cfg['games']:rows.append(dict(序号=g['index']+1,seed=g['seed'],座位=g['seat'],选局=g['rule'],u1280=summary(g,B/'baseline'),配对u2400=summary(g,B)))
 atomic_json(B/'COMPARISON.json',dict(status='RECOMPUTED',rows=rows,missing=['chain_to_old_bomb','timing_batch_threats','effective_threats','training_rewards'],distance_definition=read(B/'game_000/PASSIVE_METRICS.json')['definition']))
 path=B/cfg['html']
 if not (B/'PACKAGE.json').exists():run([sys.executable,'-m','scripts.build_bomb_threat_a3sr_replay'],env,'BUILD.log')
 html=path.read_text()
 html=html.replace('已通过用户批准的网页替换标准；不宣称逐项全面不差。','独立预览，不代表发布或整体提升。指定配对导出实际为旧A3SR c34/u2400，任务别名R1，不是新R1 c35。')
 html=html.replace('顺序前8局、最早胜负局、P1单方零泡与旧u800砖隔开同seed均按预声明规则选择。','按本任务固定首8、P1前3胜/平、零泡对照、胜平差异顺序选择；是筛选展示，不是总体胜率样本。')
 data=json.dumps(rows,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
 extra='''<style>table{border-collapse:collapse;font-size:12px;width:100%}td,th{border:1px solid #47596a;padding:5px;text-align:left}#comparison{overflow-x:auto;margin-top:10px}.summary{padding:20px;overflow-x:auto}.note{color:#edc573}</style><section class="summary"><h2>18局同seed对照摘要（u1280 → 配对u2400）</h2><p class="note">模型谱系：旧A3SR c34/u2400。链上旧泡、异时序批次威胁、有效威胁与训练即时奖励未记录，不填0。距离是当前墙/砖四邻接最短路，忽略泡、人和危险；只对双方存活且连通的帧取均值/≤3格比例，不能等同安全路径或进攻价值。</p><div id="summary-table"></div></section><script>const previewComparison=DATA;
const keys=['放泡','连上旧泡','异时序批次威胁','有效威胁','平均绕墙距离','三格内比例','距离有效帧','双活帧','不可达帧','属性终值','因果击杀','交付','自伤','胜负'];
function val(x){return typeof x==='object'?JSON.stringify(x):String(x)}function table(rows){const t=document.createElement('table');let h=t.insertRow();for(const k of ['序号 / seed / 席位',...keys]){let c=document.createElement('th');c.textContent=k;h.append(c)}for(const r of rows){let tr=t.insertRow();tr.insertCell().textContent=String(r.序号).padStart(2,'0')+' / '+r.seed+' / P'+r.座位;for(const k of keys)tr.insertCell().textContent=val(r.u1280[k])+' → '+val(r.配对u2400[k])}return t}
document.getElementById('summary-table').append(table(previewComparison));const c=document.createElement('div');c.id='comparison';document.querySelector('header').append(c);
function showCompare(){if(!window.reviewAudit)return;c.replaceChildren(table([previewComparison[Number(reviewAudit.key)]]));const root=document.getElementById('diagnostic-timeline');if(root.children.length)return;const events=docs[String(reviewAudit.index)].events.filter(e=>e.player===reviewAudit.seat);const title=document.createElement('div');title.textContent='中文事件时间轴（完整Review真实事件）';root.append(title);for(const e of events){let b=document.createElement('button');b.textContent=(e.tick/10).toFixed(1)+'秒 '+(e.kind==='bomb'?'模型放泡':'模型死亡');b.onclick=()=>seek(e.tick);root.append(b)}}new MutationObserver(showCompare).observe(document.getElementById('identity'),{childList:true});showCompare();</script>'''.replace('DATA',data)
 if 'const previewComparison=' not in html:html=html.replace('</body>',extra+'</body>');path.write_text(html)
 package=read(B/'PACKAGE.json');package.update(html_sha256=sha(path),bytes=path.stat().st_size,comparison_sha256=sha(B/'COMPARISON.json'));atomic_json(B/'PACKAGE.json',package)
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory='.'));threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  if not (B/'BROWSER.json').exists() or read(B/'BROWSER.json')['html_sha256']!=sha(path):run(['node','scripts/verify_slot00_u2400_preview_browser_20261030.js'],dict(env,REPLAY_URL=f'http://127.0.0.1:{server.server_port}/{path}',BROWSER_LOCAL='1'),'BROWSER.log')
 finally:server.shutdown()
 assert read(B/'BROWSER.json')['status']=='PASS'
 if (B/'UPLOAD_LOCATIONS.json').exists():urls=read(B/'UPLOAD_LOCATIONS.json')['urls']
 else:
  uploader=Path(shutil.which('jdme-notify')).parent.parent/'oss.sh';up=subprocess.run([str(uploader),str(path)],env=dict(os.environ,PYTHON_BIN='/usr/bin/python3'),capture_output=True,text=True,timeout=180,check=True)
  urls={}
  for line in up.stdout.splitlines():
   if 'Location:' in line:
    k,v=line.split('Location:',1);urls['internal' if 'INNER' in k else 'external']=v.strip()
  atomic_json(B/'UPLOAD_LOCATIONS.json',dict(status='UPLOADED_READBACK_PENDING',urls=urls))
 original_urls=dict(urls)
 if (B/'EXTERNAL_ROUTE_PROBE.json').exists():
  verified=[r for r in read(B/'EXTERNAL_ROUTE_PROBE.json')['results'] if r['label']=='official_alias' and r['byte_equal'] and r['sha256']==sha(path)]
  if verified:urls['external']=verified[0]['url']
 assert set(urls)=={'external','internal'}
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));reads={};raw=path.read_bytes()
 for kind,url in urls.items():
  with opener.open(url,timeout=90) as response:body=response.read();status=response.status
  assert status==200 and body==raw;reads[kind]=dict(url=url,http_status=status,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),byte_equal=True)
 atomic_json(B/'UPLOAD_VERIFIED.json',dict(status='BOTH_URLS_BYTE_VERIFIED',reads=reads,uploader_urls=original_urls,external_route='same object via official jcloudcs public endpoint; original jdcloud-oss endpoint timed out'))
 run(['node','scripts/verify_slot00_u2400_preview_browser_20261030.js'],dict(env,REPLAY_URL=urls['external']),'REMOTE_BROWSER.log')
 assert read(B/'REMOTE_BROWSER.json')['status']=='PASS'
 report=Path('reports/preview_slot00_u2400_20261030.md');lines=['# 配对slot00 u2400预览 — 20261030','','已完成18条真实生产JS Review与同seed u1280对照。指定artifact实际为旧A3SR c34，不是新R1 c35；页面明确标注。CPU两核录制，不接触训练。','',f"来源 `{cfg['model']['checkpoint']}`，policy SHA `{cfg['model']['policy_sha256']}`。",'', '链上旧泡/异时序威胁/有效威胁/训练奖励未记录；距离仅为双方存活连通时的墙砖几何路径，不代表安全或Q。', '', '|序号|seed|席位|放泡 u1280→u2400|因果击杀|交付|胜负|','|---|---|---|---|---|---|---|']
 for r in rows:lines.append('|'+ '|'.join([str(r['序号']),str(r['seed']),str(r['座位'])]+[str(r['u1280'][k])+'→'+str(r['配对u2400'][k]) for k in ['放泡','因果击杀','交付','胜负']])+'|')
 lines+=['','按任务筛选的展示集不能估计总体改进。外网/内网均HTTP200字节级一致；18条浏览器完整Review/终局校验与逐条播放通过。链接见本地回执，不向追踪文件写资产身份。',f'证据 `{B}/RECEIPT.json`、`COMPARISON.json`、`BROWSER.json`、`REMOTE_BROWSER.json`。',''];report.write_text('\n'.join(lines))
 receipt=dict(status='COMPLETED',url=urls['external'],reads=reads,model=cfg['model'],identity_correction=cfg['identity_correction'],manifest_sha256=sha(B/'MANIFEST.json'),comparison_sha256=sha(B/'COMPARISON.json'),html_sha256=sha(path),entries=18,full_game_pairs=18,missing_metrics=read(B/'COMPARISON.json')['missing'],local_browser='PASS',remote_browser='PASS',report=str(report),unix=time.time());atomic_json(B/'RECEIPT.json',receipt)
 message='slot00 u2400预览完成（指定导出实际旧A3SR c34，非新R1），18条生产JS困难Hunter/argmax完整Review，双URL字节校验与浏览器播放通过。'+urls['external']+'\n18局摘要：序号 P席位 放泡u1280→u2400 胜负\n'+'\n'.join(f"{r['序号']:02d} P{r['座位']} {r['u1280']['放泡']}→{r['配对u2400']['放泡']} {r['u1280']['胜负']}→{r['配对u2400']['胜负']}" for r in rows)+'\n完整18局对照表已附页面；威胁链缺测标未记录。训练与正式发布继续。'
 result=subprocess.run(['jdme-notify','send',message],capture_output=True,timeout=60);atomic_json(B/'NOTIFICATION.json',dict(status='SENT' if result.returncode==0 else 'FAILED',exit_code=result.returncode,unix=time.time()));assert result.returncode==0
 atomic_json(B/'TASK_CONSUMED.json',dict(status='COMPLETED',receipt_sha256=sha(B/'RECEIPT.json'),unix=time.time()))
 with Path('docs/training_experiment_ledger.md').open('a') as f:f.write('\n\n## 20261030 配对slot00 u2400独立预览\n\n指定配对导出为旧A3SR c34/u2400（任务称R1，新R1实际c35），未替换权重。18条固定选局、u1280同seed对照，CPU两核，生产JS stock-window-v1/argmax/困难Hunter/老师关闭。完整动作hash、终局与已有outcome一致，外网/内网字节hash及浏览器播放通过。缺测威胁/连锁/训练奖励标未记录。来源、分母、18局摘要见 `reports/preview_slot00_u2400_20261030.md` 和 `runs/bomb_threat_v1/replay_r1_slot00_u2400_preview/RECEIPT.json`。不以筛选展示集宣称总体提升，不改训练或发布选择。\n')
if __name__=='__main__':
 try:main()
 except Exception as e:atomic_json(B/'FINISH_FAILURE.json',dict(status='BLOCKED',error=str(e),unix=time.time()));raise
