const P=QQTReview,$=id=>document.getElementById(id);
P.registerSimulator('2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45',QQT);
const realFetch=window.fetch.bind(window);window.fetch=async function(url,options){if(embeddedJSON[String(url)])return {json:async()=>structuredClone(embeddedJSON[String(url)])};throw Error('该自包含回放不发送网络或数据库请求');};
let player,preview,renderer,doc,current,playing=false,last=0,acc=0,frames=0,readyAt=0,loadId=0,assetPromise=QQTVisual.loadAssets(replayLevel);
for(const e of entries)$('episode').add(new Option(e.label,e.key));
const fmt=t=>`${String(Math.floor(t/600)).padStart(2,'0')}:${(t%600/10).toFixed(1).padStart(4,'0')} · 第${t}帧`;
function chart(){const c=$('chart'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);g.strokeStyle='#425467';g.lineWidth=1;g.beginPath();g.moveTo(36,25);g.lineTo(36,260);g.lineTo(570,260);g.stroke();g.fillStyle='#a9bfd0';g.textAlign='center';g.font='20px sans-serif';g.fillText('未记录',300,145);g.font='14px sans-serif';g.fillText('局内时间（秒）',310,287);g.textAlign='left';g.fillText('0',36,279);g.textAlign='right';g.fillText(String(doc.actions.length/10),570,279);}
function draw(){if(!player||!renderer)return;preview.seek(Math.min(player.sim.t+1,doc.actions.length));const row=doc.actions[player.sim.t]||[4,0,4,0],m=QQTReviewView.motion(player.sim,preview.sim,acc/100,true,doc.events,[row[0],row[2]]);
 m.characters=['maomao','maomao'];m.playerPalettes=[0,1].map(p=>p===current.seat?null:'blue');m.playerLabels=[0,1].map(p=>`${p===current.seat?'模型毛毛':'蓝色 Hunter'} P${p}`);m.humanPid=current.seat;m.hideResult=true;
 renderer.render(player.sim,player.sim.t*100,m);$('seek').value=player.sim.t;$('clock').textContent=fmt(player.sim.t)+' / '+fmt(doc.actions.length);$('play').textContent=playing?'暂停':'播放';frames++;}
function seek(t){playing=false;acc=0;renderer.reset();player.seek(Math.max(0,Math.min(t,doc.actions.length)));draw();}
function diagnosticTimeline(){const root=$('diagnostic-timeline');root.replaceChildren();const marks=current.annotations||[];if(!marks.length)return;
 const title=document.createElement('div');title.textContent='P0振荡/持包停滞诊断时间轴（点击跳转；区段右端不含）';root.append(title);
 for(const mark of marks){const b=document.createElement('button');b.textContent=`${mark.label}：${(mark.start/10).toFixed(1)}–${(mark.end/10).toFixed(1)}秒（${mark.start}–${mark.end}帧）`;b.style.margin='3px';b.dataset.start=mark.start;b.dataset.end=mark.end;b.onclick=()=>seek(mark.start);root.append(b);}
}
async function load(){const version=++loadId;playing=false;window.reviewAudit=null;$('status').textContent='正在验证完整动作、状态hash与终局…';const entry=entries.find(e=>e.key===$('episode').value),nextDoc=docs[String(entry.index)];
 const nextPlayer=new P.Player(nextDoc,replayLevel);await nextPlayer.verify();const assets=await assetPromise;if(version!==loadId)return;
 current=entry;doc=nextDoc;player=nextPlayer;preview=new P.Player(doc,replayLevel);preview.cache=player.cache;renderer=QQTVisual.createRenderer($('game'),replayLevel,assets);acc=0;frames=0;readyAt=performance.now();$('seek').max=doc.actions.length;
 $('identity').textContent=`checkpoint ${doc.meta.checkpoint}；policy SHA256 ${doc.meta.checkpoint_sha256}；model JSON SHA256 ${doc.meta.model_sha256}；seed ${entry.seed}；P${entry.seat}；规则选局：${entry.selection}；采集时间 ${doc.meta.recorded_at}`;
 $('status').textContent=(entry.matched?'完整Review核验通过；与fresh256的胜负、放泡、击杀及终局逐项一致。':'重放不一致；原fresh256记录保留，不替换。')+' 本局 '+entry.label;
 diagnosticTimeline();chart();draw();window.reviewAudit={key:entry.key,index:entry.index,seat:entry.seat,verified:true,matched:entry.matched,policy:doc.meta.checkpoint_sha256,model:doc.meta.model_sha256,teacher:doc.meta.teacher,annotations:entry.annotations||[],get tick(){return player.sim.t},get frames(){return frames},get fps(){return frames*1000/(performance.now()-readyAt)},get labels(){return [0,1].map(p=>p===current.seat?'model/maomao':'Hunter/blue')}};
}
$('episode').onchange=()=>load().catch(fail);$('play').onclick=()=>{if(!player)return;if(player.sim.t===doc.actions.length)seek(0);playing=!playing;last=performance.now();draw();};$('restart').onclick=()=>seek(0);$('back').onclick=()=>seek(player.sim.t-1);$('next').onclick=()=>seek(player.sim.t+1);$('seek').oninput=()=>seek(Number($('seek').value));
function fail(e){$('status').textContent='核验失败：'+e.message;console.error(e);}
function frame(now){if(playing&&player){acc+=Math.min(250,now-last)*Number($('speed').value);while(acc>=100){acc-=100;if(!player.step()){playing=false;acc=0;break;}if(player.info)renderer.addExplosion(player.info,player.sim.t*100);}draw();}last=now;requestAnimationFrame(frame);}requestAnimationFrame(frame);load().catch(fail);
