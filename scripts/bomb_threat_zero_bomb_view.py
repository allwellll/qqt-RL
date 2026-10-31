"""Build a local diagnostic viewer using the production Review Player/rendering."""
import json
from pathlib import Path

OUT = Path('runs/bomb_threat_v1/iter2/zero_bomb_audit')


def main():
    complete = json.loads((OUT/'COMPLETE.json').read_text())
    entries = []
    for item in complete['results']:
        audit = json.loads(Path(item['audit']).read_text())
        entries.append(dict(key=item['key'], file=item['replay'], audit=audit))
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<base href="../../../../web/"><title>零放泡审计 · 生产 Review</title>
<style>body{font:16px sans-serif;margin:20px;background:#eee}main{display:flex;gap:20px;align-items:flex-start}canvas{width:600px;max-width:65vw}aside{max-width:550px}button,select,input{margin:5px}pre{white-space:pre-wrap}li{margin:8px}</style>
<h1>零放泡审计 · 固定 seed 完整对局</h1><p>模型 P0 毛毛 · 对手 P1 蓝色毛毛；原生带砖 JS，老师关闭，greedy；不发布模型，不连接数据库。双方零泡是统计分类，不等同永久不可达。</p>
<select id="episode"></select><button id="play">播放</button><select id="speed"><option>1</option><option>4</option><option>16</option></select><input id="seek" type="range" min="0" max="2400" value="0"><span id="clock"></span><p id="status">验证中</p>
<main><canvas id="game" width="900" height="810"></canvas><aside><h2>中文时间轴 / 规则证据</h2><ul id="timeline"></ul><pre id="identity"></pre></aside></main>
<script src="sim.js"></script><script src="tactical_arena.js"></script><script src="review_dagger.js"></script><script src="review_advantage.js"></script><script src="review_route.js"></script><script src="review_protocol.js"></script><script src="review_presentation.js"></script><script src="visual_renderer.js"></script><script src="sim-stock-window-v1.js"></script>
<script>
const entries='''+json.dumps(entries, ensure_ascii=False)+''';
const P=QQTReview,$=id=>document.getElementById(id);let player,preview,doc,renderer,playing=false,last=0,acc=0,frames=0,loadedAt;
P.registerSimulator('2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45',QQT);
for(const e of entries)$('episode').add(new Option(e.key,e.key));
const clock=t=>`${(t/10).toFixed(1)}秒 / tick ${t}`;
function draw(){preview.seek(Math.min(player.sim.t+1,doc.actions.length));const row=doc.actions[player.sim.t]||[4,0,4,0],motion=QQTReviewView.motion(player.sim,preview.sim,acc/100,true,doc.events,[row[0],row[2]]);renderer.render(player.sim,player.sim.t*100,motion);$('seek').value=player.sim.t;$('clock').textContent=clock(player.sim.t);frames++;}
async function load(){playing=false;const entry=entries.find(e=>e.key===$('episode').value);doc=await(await fetch('../'+entry.file)).json();const level=(await(await fetch('assets/maps/levels.json')).json()).find(x=>x.qqt_id===806);
 player=new P.Player(doc,level);await player.verify();preview=new P.Player(doc,level);preview.cache=player.cache;renderer=QQTVisual.createRenderer($('game'),level,await QQTVisual.loadAssets(level));
 $('seek').max=doc.actions.length;acc=0;$('status').textContent='Review完整hash与终局已验证 · '+(doc.summary.winner===null?'平局':doc.summary.winner===0?'模型胜':'对手胜');
 $('identity').textContent=JSON.stringify({checkpoint:doc.meta.checkpoint,policy_sha256:doc.meta.checkpoint_sha256,model_sha256:doc.meta.model_sha256,seed:doc.meta.seed,opponent:doc.meta.opponent,engine:'production JS stock-window-v1',teacher:false,requests:entry.audit.requests,actual_placed:entry.audit.placed,rejection:entry.audit.rejects},null,2);
 const notes=[{tick:0,note:`冻结出生 ${JSON.stringify(doc.meta.frozen_birth.cells)}；当前砖块下连通=${entry.audit.open_connected}，忽略可炸砖后连通=${entry.audit.wall_only_connected}`}];
 for(const e of doc.events)notes.push({tick:e.tick,note:`${e.player===0?'模型毛毛':'蓝色Hunter'}：${e.kind==='bomb'?'实际放泡':e.kind==='death'?'死亡':e.kind}`});
 if(entry.key.startsWith('A2')){const trace=await(await fetch('../'+entry.file.replace('review.json','trace.json'))).json();for(const reason of ['dead','native_sugar']){const hits=trace.filter(r=>r.placement[0].requested&&r.placement[0].why.includes(reason));if(hits.length)notes.push({tick:hits[0].tick,note:`请求被拒：${reason==='dead'?'角色已死亡':'原生糖浆控制'}，共${hits.length}次，首${hits[0].tick}/末${hits.at(-1).tick} tick`});}}
 notes.push({tick:doc.summary.ticks,note:`完整终局；实际放泡 ${entry.audit.placed.join('/')}，死亡 ${entry.audit.deaths.join('/')}`});$('timeline').replaceChildren();for(const e of notes.sort((a,b)=>a.tick-b.tick)){const li=document.createElement('li'),b=document.createElement('button');b.textContent=clock(e.tick);b.onclick=()=>{playing=false;acc=0;player.seek(e.tick);draw();};li.append(b,document.createTextNode(e.note));$('timeline').append(li);}
 frames=0;loadedAt=performance.now();draw();window.reviewAudit={loaded:true,key:entry.key,verified:true,get tick(){return player.sim.t;},get frames(){return frames;},get fps(){return frames*1000/(performance.now()-loadedAt);}};
}
$('episode').onchange=()=>load().catch(fail);$('play').onclick=()=>{if(player.sim.t===doc.actions.length)player.seek(0);playing=!playing;last=performance.now();};$('seek').oninput=()=>{playing=false;acc=0;player.seek(Number($('seek').value));draw();};
function fail(e){$('status').textContent=e.stack;console.error(e);}
function frame(now){if(playing&&player){acc+=Math.min(250,now-last)*Number($('speed').value);while(acc>=100){acc-=100;if(!player.step()){playing=false;acc=0;break;}}draw();}last=now;requestAnimationFrame(frame);}requestAnimationFrame(frame);load().catch(fail);
</script></html>''')


if __name__ == '__main__':
    main()
