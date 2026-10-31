"""Package only verified Review recordings and renderer assets into one HTML."""
import base64,json,mimetypes
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
B=Path('runs/bomb_threat_v1/replay_best_u800')

def main():
    cfg=json.loads((B/'MANIFEST.json').read_text());entries=[];docs={};checks=[]
    for k,item in enumerate(cfg['games']):
        out=B/f"game_{item['index']:03d}";d=json.loads((out/'review.json').read_text());m=json.loads((out/'MATCH.json').read_text());checks.append(m)
        assert m['review_verified'] and sha(out/'review.json')==m['review_sha256']
        docs[str(item['index'])]=d
        r=item['expected'];seat=item['seat'];result='平' if r['winner'] in (None,-1) else '胜' if r['winner']==seat else '负'
        entries.append(dict(key=str(k),index=item['index'],seed=item['seed'],seat=seat,selection=item['rule'],annotations=item.get('annotations',[]),matched=m['status']=='MATCH',
            label=f"{k+1:02d} · index{item['index']} · P{seat} · {result} · 击杀{r['causal_kills'][seat]} · 放泡{r['placed'][seat]} · "+('顺序前8局' if k<8 else 'P1零泡聚焦（重复）' if k<10 else item['rule'])))
    levels=json.loads(Path('web/assets/maps/levels.json').read_text());level=next(x for x in levels if x['qqt_id']==806)
    names=['assets/maps/elements.json','assets/item/items.json','assets/native/sprites.json'];jsonassets={n:json.loads((Path('web')/n).read_text()) for n in names}
    el,items,native=(jsonassets[n] for n in names)
    ids={abs(v) for l in level['layers'] for v in l if v}
    files={level.get('bg','assets/bg/抢包子.png'),'assets/角色4×4精灵图.png','assets/角色c4×4.png','assets/bomb-custom/经典黄泡泡.png','assets/shadow.png','assets/point.png'}
    files.update(f'assets/flame/flame_{d}_{f}.png' for d in 'UDLR' for f in range(1,7));files.update(f'assets/flame/flame_C_{f}.png' for f in (1,2))
    files.update(m['file'] for m in items.values());files.update(el[str(i)]['file'] for i in ids if str(i) in el)
    files.update(m['file'] for actor in native['actors'].values() for m in actor['actions'].values());files.update(m['file'] for m in native['effects'].values())
    assets={n:'data:'+str(mimetypes.guess_type(n)[0] or 'image/png')+';base64,'+base64.b64encode((Path('web')/n).read_bytes()).decode() for n in sorted(files)}
    dump=lambda v:json.dumps(v,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    header='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A3 u800 对局录像</title><link rel="icon" href="data:,"><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111820;color:#e6edf3;font:15px system-ui,sans-serif}header{padding:16px 24px;border-bottom:1px solid #35414d}h1{font-size:21px;margin:0 0 10px}select,button{background:#26333e;color:#fff;border:1px solid #566879;border-radius:7px;padding:9px}#episode{width:min(100%,1000px)}.meta{font-size:12px;color:#b7c7d6;margin-top:8px;overflow-wrap:anywhere}main{display:grid;grid-template-columns:minmax(540px,1.6fr) minmax(300px,1fr);gap:20px;padding:20px;max-width:1580px;margin:auto}.board{background:#1c2630;border-radius:10px;overflow:hidden}#game{display:block;width:100%;height:auto;max-height:72vh;object-fit:contain}.transport{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:12px 0}#seek{width:100%;accent-color:#58b4e8}#clock{font-variant-numeric:tabular-nums}aside{padding:18px;background:#1c2630;border-radius:10px;align-self:start}h2{font-size:16px;margin:0 0 14px}#reward{font-size:28px;margin:15px 0 28px}#chart{width:100%;height:260px;background:#15202a;border:1px solid #3b4b5a;border-radius:7px}#chart-label{color:#b7c7d6;font-size:13px}.timeline{margin-top:8px;color:#adbfce}#status{font-size:13px;padding-top:8px;color:#b7c7d6}@media(max-width:900px){main{grid-template-columns:1fr;padding:10px}.transport{gap:5px}header{padding:12px}#game{max-height:none}}
</style></head><body><header><h1>A3 u800 · 困难 Hunter 完整对局录像</h1><select id="episode" aria-label="选择对局"></select><div class="meta">候选，B fresh256 未通过、未晋升。生产 JS · stock-window-v1 · 带砖自然局 · argmax · 老师关闭。10条入口 / 8个独立对局；最后2条为按规则重复聚焦的 P1 单方零泡局。</div><details class="meta"><summary>核验模型与选局来源</summary><div id="identity"></div></details></header><main><section><div class="board"><canvas id="game" width="900" height="810" aria-label="生产Review对局画面"></canvas></div><div class="transport"><button id="restart">从头</button><button id="back">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><select id="speed" aria-label="播放速度"><option value=".5">0.5倍</option><option value="1" selected>1倍</option><option value="2">2倍</option><option value="4">4倍</option><option value="16">16倍</option></select><span id="clock">00:00.0</span><input id="seek" type="range" min="0" max="2400" value="0" aria-label="局内时间帧横轴"></div><div class="timeline">时间轴：局内秒数 / 原生帧（每帧100毫秒）</div><div id="status">正在核验完整回放…</div></section><aside><h2>当前帧即时奖励</h2><div id="reward">未记录</div><h2>奖励走势</h2><canvas id="chart" width="600" height="300" aria-label="奖励走势，未记录"></canvas><div id="chart-label">未记录；不以 Value、Logit 或零值代替奖励。</div></aside></main>'''
    header=header.replace('10条入口 / 8个独立对局；最后2条为按规则重复聚焦的 P1 单方零泡局。',f"{len(entries)}条入口 / {len(docs)}个独立对局；第9、10条为重复聚焦的 P1 单方零泡局；新增胜负局及诊断代表均按预声明规则选择，入口标签注明来源。")
    header=header.replace('<div id="status">','<div id="diagnostic-timeline" aria-label="中文振荡停滞时间轴"></div><div id="status">')
    # Pure SHA fallback keeps exact Review checks on HTTP OSS as well as HTTPS.
    sha_js='''function reviewDigest(input){const b=Array.from(input),n=b.length*8,H=[],K=[];for(let x=2;K.length<64;x++){let prime=true;for(let f=2;f*f<=x;f++)if(x%f===0){prime=false;break;}if(prime){if(H.length<8)H.push((Math.sqrt(x)%1*4294967296)|0);K.push((Math.cbrt(x)%1*4294967296)|0);}}b.push(128);while(b.length%64!==56)b.push(0);for(let i=7;i>=0;i--)b.push(i>=4?0:(n>>>i*8)&255);const R=(x,n)=>(x>>>n)|(x<<(32-n));for(let o=0;o<b.length;o+=64){const w=[];for(let t=0;t<64;t++){if(t<16)w[t]=(b[o+t*4]<<24)|(b[o+t*4+1]<<16)|(b[o+t*4+2]<<8)|b[o+t*4+3];else{let a=w[t-15],z=w[t-2];w[t]=(w[t-16]+(R(a,7)^R(a,18)^(a>>>3))+w[t-7]+(R(z,17)^R(z,19)^(z>>>10)))|0;}}let [a,c,d,e,f,g,h,i]=H;for(let t=0;t<64;t++){let q=(i+(R(f,6)^R(f,11)^R(f,25))+((f&g)^(~f&h))+K[t]+w[t])|0,z=((R(a,2)^R(a,13)^R(a,22))+((a&c)^(a&d)^(c&d)))|0;i=h;h=g;g=f;f=(e+q)|0;e=d;d=c;c=a;a=(q+z)|0;}[a,c,d,e,f,g,h,i].forEach((x,j)=>H[j]=(H[j]+x)|0);}return H.map(x=>(x>>>0).toString(16).padStart(8,'0')).join('');}'''
    scripts=[sha_js,'const embeddedAssets='+dump(assets)+';const embeddedJSON='+dump(jsonassets)+';const entries='+dump(entries)+';const docs='+dump(docs)+';const replayLevel='+dump(level)+';']
    sources=['sim.js','tactical_arena.js','review_dagger.js','review_advantage.js','review_route.js','review_protocol.js','review_presentation.js','visual_renderer.js','sim-stock-window-v1.js']
    for name in sources:
        s=(Path('web')/name).read_text()
        if name=='review_protocol.js':
            a=s.index('  async function sha256(bytes)');z=s.index('  const hash =',a);s=s[:a]+'  async function sha256(bytes) { return reviewDigest(bytes); }\n'+s[z:]
        if name=='visual_renderer.js':s=s.replace('image.src = src;', 'image.src = embeddedAssets[src] || src;')
        scripts.append(s)
    scripts.append(Path('scripts/bomb_threat_best_replay_page.js').read_text())
    text=header+''.join('<script>'+s.replace('</script','<\\/script')+'</script>' for s in scripts)+'</body></html>'
    dest=B/'20261029-a3-u800-review.html';dest.write_text(text)
    atomic_json(B/'PACKAGE.json',dict(status='PACKAGED',html=str(dest),html_sha256=sha(dest),bytes=dest.stat().st_size,entries=entries,
        matches={str(m['index']):m['status'] for m in checks},assets={f:sha(Path('web')/f) for f in sorted(files)},source_sha256={str(Path('web')/f):sha(Path('web')/f) for f in sources},external_dependencies=0,models_uploaded=False))
    print(str(dest),dest.stat().st_size)

if __name__=='__main__':main()
