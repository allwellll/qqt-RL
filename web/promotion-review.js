'use strict';
(async()=>{
 const $=id=>document.getElementById(id),P=QQTFiveSlotsReview;
 let manifest,player,renderer,playing=false,last=0,accumulator=0,loadVersion=0;
 const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
 function draw(){
  if(!player)return;const s=player.sim,d=player.document,a=d.actions[s.t];
  renderer.render(s,s.t*100,{intents:a?[a[0],a[2]]:[4,4],humanPid:-1,
   prevPos:s.pos,curPos:s.pos,lastTickT:s.t*100,tickMs:100});$('seek').value=s.t;
  $('state').textContent=`${(s.t/10).toFixed(1)} / ${(d.actions.length/10).toFixed(1)} 秒 · 模型 P${d.meta.learner_seat} · ${s.done?(s.winner===null?'平局':`P${s.winner} 获胜`):'对局进行中'}`;
  $('decision').textContent=[0,1].map(p=>`P${p} 库存：${s.itemSlots[p].map((x,i)=>`槽${i+1} ${x.item===1?'香蕉皮':x.item===2?'慢慢胶':'道具'+x.item}×${x.count}`).join('；')||'空'}\n下一动作：${a?`移动${a[p*2]} / ${QQTFiveSlots.ACTION_NAMES[a[p*2+1]]}`:'终局'}`).join('\n\n');
 }
 async function load(index){
  const version=++loadVersion;playing=false;$('play').disabled=$('restart').disabled=$('seek').disabled=true;$('notice').textContent='正在校验完整录像…';
  const row=manifest.replays[index],bytes=await(await fetch(row.file)).arrayBuffer();if(await digest(bytes)!==row.sha256)throw Error('录像 SHA-256 不一致');
  const doc=JSON.parse(new TextDecoder().decode(bytes));if(doc.meta.checkpoint_sha256!==manifest.policy_sha256||doc.meta.model_sha256!==manifest.model_sha256||doc.meta.simulator_sha256!==manifest.rule_sha256)throw Error('录像与模型/规则不一致');
  const level=JSON.parse(await(await fetch('assets/maps/levels.json')).text()).find(x=>x.qqt_id===806);const next=new P.Player(doc,level);await next.verify();const assets=await QQTVisual.loadAssets(level);if(version!==loadVersion)return;
  player=next;renderer=QQTVisual.createRenderer($('game'),level,assets);$('seek').max=doc.actions.length;$('play').disabled=$('restart').disabled=$('seek').disabled=false;$('play').textContent='播放';
  $('notice').textContent='文件 SHA、完整状态检查点与终局校验通过';$('identity').textContent=`${manifest.display_name} · seed ${doc.meta.seed} · 模型 P${doc.meta.learner_seat}\nPolicy SHA: ${manifest.policy_sha256}\n模型 SHA: ${manifest.model_sha256}\n规则 SHA: ${manifest.rule_sha256}`;
  window.promotionReview={player,manifest,verified:true};draw();
 }
 try{
  manifest=await(await fetch('promotion-review.json')).json();const bytes=await(await fetch('sim-bun-native-v2.js')).arrayBuffer();if(await digest(bytes)!==manifest.rule_sha256)throw Error('线上规则 SHA 不一致');
  $('title').textContent=manifest.display_name+' · 固定8局完整 Review';manifest.replays.forEach((r,i)=>$('recording').add(new Option(`seed ${r.seed} · 模型 P${r.seat}`,i)));
  $('recording').onchange=()=>load(+$('recording').value).catch(fail);
  $('play').onclick=()=>{if(player.sim.t===player.document.actions.length)player.seek(0);playing=!playing;last=performance.now();accumulator=0;$('play').textContent=playing?'暂停':'播放';draw();};
  $('restart').onclick=()=>{playing=false;player.seek(0);renderer.reset();$('play').textContent='播放';draw();};
  $('seek').oninput=()=>{playing=false;player.seek(+$('seek').value);renderer.reset();$('play').textContent='播放';draw();};
  function frame(now){if(playing){accumulator+=Math.min(250,now-last)*+$('speed').value;while(accumulator>=100){accumulator-=100;if(!player.step()||player.sim.t===player.document.actions.length){playing=false;$('play').textContent='播放';break;}if(player.info)renderer.addExplosion(player.info,player.sim.t*100);}draw();}last=now;requestAnimationFrame(frame);}
  await load(0);requestAnimationFrame(frame);
 }catch(e){fail(e);}
 function fail(e){$('notice').textContent='校验失败：'+e.message;console.error(e);}
})().catch(console.error);
