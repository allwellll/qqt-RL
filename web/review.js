'use strict';
(async function() {
  const P=QQTReview,$=id=>document.getElementById(id);
  const auth=QQTDeviceAuth.create(localStorage,crypto);
  QQTPanels.mount(document,{auth});
  if(auth.authorized())$('advanced-open').click();
  const labels={enemy_drop_unpicked:'掉包后未拾',window_closed_empty:'窗口未切换',return_deviation:'返程偏航',cage_no_delivery:'到笼未交付',self_bomb:'自炸',stagnation:'停滞',bomb:'放泡',use_item:'使用道具',death:'死亡',revive:'复活',control:'进入糖泡',control_end:'解除控制',drop:'掉包',native_transfer:'接触转移',pickup:'拾取',stock_theft:'偷包',delivery:'交付',phase:'阶段切换'};
  let catalog,doc,player,renderer,playing=false,accumulator=0,last=performance.now(),rows=[],editing=null,loadToken=0;
  const text=(tag,value,className)=>{const e=document.createElement(tag);e.textContent=value;if(className)e.className=className;return e;};
  const say=value=>$('notice').textContent=value;
  const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
  const number=n=>Math.abs(n)<1e-8?'0':n.toFixed(3);
  function hashPosition(){if(doc)history.replaceState(null,'',`#replay=${encodeURIComponent(doc.meta.id)}&tick=${player.sim.t}`);}
  function pause(){playing=false;accumulator=0;update();}
  function seek(tick){if(!player)return;playing=false;accumulator=0;renderer.reset();player.seek(tick);const info=player.info;if(info)renderer.addExplosion(info,tick*100);update();hashPosition();draw();}
  function list(){
    const filter=$('filter').value.toLowerCase();$('replay-list').replaceChildren();
    for(const r of catalog.replays){
      if(!`${r.id} ${r.summary.failures.map(k=>labels[k]).join(' ')}`.toLowerCase().includes(filter))continue;
      const b=text('button','',`replay-row${doc?.meta.id===r.id?' active':''}`);b.dataset.replay=r.id;
      b.append(text('strong',`H7 g32 · ${r.variant==='open'?'清砖图':'原始图'}`),text('small',`seed ${r.seed} · ${P.time(r.summary.ticks)} · ${r.summary.winner===null?'平局':r.summary.winner===0?'模型胜':'对手胜'}`));
      const tags=text('div','','tags');for(const k of r.summary.failures)tags.append(text('span',labels[k]||k,'tag'));b.append(tags);b.onclick=()=>load(r.id,0).catch(fail);$('replay-list').append(b);
    }
  }
  function details(target,values){const dl=document.createElement('dl');for(const [key,value] of values){const div=document.createElement('div');div.append(text('dt',key),text('dd',value));dl.append(div);}target.replaceChildren(dl);}
  function currentReward(tick){return doc.rewards.find(r=>r[0]===tick)?.[Number($('reward-player').value)+1]||{};}
  function rewardValues(){const all={};for(const row of doc.rewards){if(row[0]>player.sim.t)break;for(const [key,v] of Object.entries(row[Number($('reward-player').value)+1]))all[key]=(all[key]||0)+v;}return all;}
  function update(){
    if(!player)return;const s=player.sim,t=s.t;
    $('exact-time').textContent=`tick ${t} · ${P.time(t)}`;$('play-state').textContent=playing?'播放中':'已暂停';$('position-panel').classList.toggle('playing',playing);
    $('locator').value=P.locator(doc,t);$('seek').value=t;$('playhead').style.left=`${t/doc.actions.length*100}%`;
    $('play').title=playing?'暂停':'播放';$('play').setAttribute('aria-label',playing?'暂停':'播放');$('play-icon').src=`assets/review-icons/${playing?'pause':'play'}.svg`;
    const delta=currentReward(t),total=rewardValues();$('reward-delta').textContent=number(sum(delta));$('reward-total').textContent=number(sum(total));
    $('components').replaceChildren();for(const k of componentNames()){const e=text('div','','component'),v=delta[k]||0;e.append(text('span',k),text('span',`${number(v)} / ${number(total[k]||0)}`,v<0?'negative':'positive'));$('components').append(e);}
    const a=t?doc.actions[t-1]:null;
    details($('state-details'),[['模型阶段',P.phase(s)],['对手阶段',P.phase(s,1)],['动作 [move, ability]',a?`${a[0]},${a[1]} | ${a[2]},${a[3]}`:'开局'],['包归属（模型 / 对手）',s.bunCarried.map(v=>v<0?'空手':`源队 ${v}`).join(' / ')],['糖泡剩余 tick',s.trapped.join(' / ')],['死亡 / 复活倒计时',s.bunRespawn.join(' / ')],['坐标 [row,col]',`${s.pos[0].toFixed(2)},${s.pos[1].toFixed(2)} | ${s.pos[2].toFixed(2)},${s.pos[3].toFixed(2)}`],['笼内存包',s.bunStored.map(a=>a.join(',')).join(' / ')],['终局',s.done?`胜者 ${s.winner===null?'平局':s.winner}`:'进行中']]);
    $('current-events').replaceChildren();const events=doc.events.filter(e=>e.tick===t);for(const e of events)$('current-events').append(text('div',`${labels[e.kind]||e.kind} · P${e.player}${e.origin===undefined?'':` · 源队${e.origin}`}${e.to?` → ${e.to}`:''}`,'event-line'));if(!events.length)$('current-events').append(text('span','—','empty'));
    chart();
  }
  function componentNames(){return doc.meta.reward_components || [...new Set(doc.rewards.flatMap(r=>Object.keys(r[1])))].sort();}
  function chart(){
    if(!doc)return;const c=$('reward-chart'),g=c.getContext('2d'),w=c.width,h=c.height,component=$('curve-component').value,mode=$('curve-mode').value;
    const values=new Array(doc.actions.length+1).fill(0);let total=0;for(let t=0;t<values.length;t++){const r=currentReward(t),v=component==='total'?sum(r):r[component]||0;total+=v;values[t]=mode==='cumulative'?total:v;}
    const lo=Math.min(0,...values),hi=Math.max(1,...values),y=v=>h-15-(v-lo)/(hi-lo)*(h-30);g.clearRect(0,0,w,h);g.strokeStyle='#343b46';g.beginPath();g.moveTo(0,y(0));g.lineTo(w,y(0));g.stroke();g.fillStyle='#8492a5';g.font='18px monospace';g.fillText(hi.toFixed(1),5,18);g.fillText(lo.toFixed(1),5,h-3);g.strokeStyle='#74cdb2';g.lineWidth=2;g.beginPath();values.forEach((v,t)=>t?g.lineTo(t/(values.length-1)*w,y(v)):g.moveTo(0,y(v)));g.stroke();g.strokeStyle='#e9f7ef';const x=player.sim.t/doc.actions.length*w;g.beginPath();g.moveTo(x,0);g.lineTo(x,h);g.stroke();
  }
  function marker(track,tick,color,title,width){const b=text('button','','mark'+(width?' phase-mark':''));b.style.left=`${tick/doc.actions.length*100}%`;b.style.background=color;b.title=`tick ${tick} · ${P.time(tick)} · ${title}`;b.setAttribute('aria-label',b.title);if(width)b.style.width=`${width/doc.actions.length*100}%`;b.onclick=e=>{e.stopPropagation();seek(tick);};$(track).append(b);}
  function timeline(){
    for(const id of ['ruler','phase-track','event-track','reward-track','annotation-track'])$(id).replaceChildren();
    for(let t=0;t<=doc.actions.length;t+=200){const e=text('span',P.time(t).slice(0,5));e.style.left=`${t/doc.actions.length*100}%`;$('ruler').append(e);}
    let start=0,phase='战斗';for(const e of doc.events.filter(e=>e.kind==='phase'&&e.player===0)){marker('phase-track',start,phase==='持包返程'?'#77c6aa':phase==='运输窗口'?'#d4ab64':'#5ba6ba',phase,e.tick-start);start=e.tick;phase=e.to;}marker('phase-track',start,'#5ba6ba',phase,doc.actions.length-start);
    for(const e of doc.events.filter(e=>e.kind!=='phase'&&e.kind!=='bomb'))marker('event-track',e.tick,'#d8ad70',labels[e.kind]||e.kind);
    for(const r of doc.rewards)if(Object.keys(r[1]).length)marker('reward-track',r[0],sum(r[1])<0?'#e78f94':'#7bcaac',`Reward Δ ${number(sum(r[1]))}`);
    for(const a of rows)marker('annotation-track',a.tick,'#e192a6',a.note||a.category);
    $('timeline').style.width=`${Number($('zoom').value)*100}%`;
  }
  const storageKey=()=>`qqt.annotations.v1.${doc.meta.id}.${doc.asset_sha256}`;
  function persist(){try{localStorage.setItem(storageKey(),JSON.stringify(P.annotationDocument(doc,rows)));say('标注已保存在本设备');}catch(_){say('存储不可用；请导出标注');}annotations();timeline();}
  function annotations(){
    $('annotations').replaceChildren();$('annotation-count').textContent=rows.length;
    for(const a of [...rows].sort((a,b)=>a.tick-b.tick)){
      const e=text('div','','annotation-row'),jump=text('button',`t${a.tick}`,'jump');jump.onclick=()=>seek(a.tick);const content=text('div','','annotation-text');content.append(text('small',`${a.category} · ${a.severity} · ${P.time(a.tick)}`),text('p',a.note));
      const edit=text('button',''),del=text('button','');for(const [b,icon,title] of [[edit,'pencil','编辑标注'],[del,'trash-2','删除标注']]){b.title=title;b.setAttribute('aria-label',title);const img=document.createElement('img');img.src=`assets/review-icons/${icon}.svg`;img.alt='';b.append(img);}edit.onclick=()=>form(a);del.onclick=()=>{rows=rows.filter(x=>x.id!==a.id);persist();};e.append(jump,content,edit,del);$('annotations').append(e);
    }
    if(!rows.length)$('annotations').append(text('div','暂无标注','empty'));
  }
  function form(a){pause();editing=a?.id||null;$('annotation-tick').value=a?.tick??player.sim.t;$('annotation-tick').max=doc.actions.length;$('annotation-category').value=a?.category||'other';$('annotation-severity').value=a?.severity||'medium';$('annotation-note').value=a?.note||'';$('annotation-form').hidden=false;$('annotation-note').focus();}
  function draw(){if(renderer&&player)renderer.render(player.sim,player.sim.t*100,{humanPid:-1,tickMs:100,lastTickT:player.sim.t*100,prevPos:player.sim.pos,curPos:player.sim.pos,characters:['pipi','maomao'],intents:doc.actions[player.sim.t-1]?[doc.actions[player.sim.t-1][0],doc.actions[player.sim.t-1][2]]:[4,4]});}
  async function load(id,tick){
    const token=++loadToken;playing=false;$('canvas-loading').hidden=false;$('canvas-loading').textContent='正在验证录像与状态 hash';$('integrity').textContent='验证中';
    const row=catalog.replays.find(r=>r.id===id);if(!row)throw Error('未知 replay ID');
    const response=await fetch(`review-replays/${row.file}`);if(!response.ok)throw Error(`录像 HTTP ${response.status}`);const bytes=new Uint8Array(await response.arrayBuffer());if(bytes.length!==row.bytes||await P.sha256(bytes)!==row.sha256)throw Error('录像文件 hash / 大小不匹配');
    const next=P.validate(JSON.parse(new TextDecoder().decode(bytes)));const mapResponse=await fetch('assets/maps/levels.json');if(!mapResponse.ok)throw Error('地图 HTTP');const level=(await mapResponse.json()).find(x=>x.qqt_id===806);if(await P.hash(level)!==next.meta.map_sha256)throw Error('地图 hash');
    const simBytes=await(await fetch('sim.js')).arrayBuffer();if(await P.sha256(simBytes)!==next.meta.simulator_sha256)throw Error('模拟器版本不匹配');
    const nextPlayer=new P.Player(next,level);await nextPlayer.verify();const assets=await QQTVisual.loadAssets(level);if(token!==loadToken)return;
    Object.defineProperty(next,'asset_sha256',{value:row.sha256});
    doc=next;player=nextPlayer;renderer=QQTVisual.createRenderer($('game'),level,assets);rows=[];editing=null;$('annotation-form').hidden=true;
    try{const saved=localStorage.getItem(storageKey());if(saved)rows=P.validateAnnotations(JSON.parse(saved),doc);}catch(_){say('已有标注无效或存储不可用');}
    $('episode-title').textContent=`${doc.meta.model} · ${doc.meta.variant==='open'?'806 清砖图':'806 原始图'}`;$('episode-meta').textContent=`${doc.meta.opponent} · seed ${doc.meta.seed} · checkpoint ${doc.meta.checkpoint_sha256.slice(0,12)}`;$('terminal').textContent=`终局：${doc.summary.winner===null?'平局':doc.summary.winner===0?'模型胜':'对手胜'}`;
    $('seek').max=doc.actions.length;$('duration').textContent=P.time(doc.actions.length);$('integrity').textContent='hash 已验证';$('integrity').style.color='#82d2b6';
    $('curve-component').replaceChildren(new Option('总 Reward','total'));for(const k of componentNames())$('curve-component').add(new Option(k,k));
    details($('teacher'),['eligible','query','accept','reject'].map(k=>[k,doc.meta.teacher?.[k]??'未测量']));
    $('provenance').replaceChildren();for(const [k,v] of [['Episode',doc.meta.episode_id],['Checkpoint SHA',doc.meta.checkpoint_sha256],['Model SHA',doc.meta.model_sha256],['Simulator SHA',doc.meta.simulator_sha256],['Opponent SHA',doc.meta.opponent_sha256],['Replay SHA',row.sha256],['Reward',doc.meta.reward_source||'未测量'],['原始 / gzip',`${row.bytes} / ${row.gzip_bytes} bytes`]])$('provenance').append(text('div',`${k}: ${v}`));
    $('canvas-loading').hidden=true;list();timeline();annotations();seek(Math.min(doc.actions.length,Math.max(0,Number.isInteger(tick)?tick:0)));say('录像与间隔状态 hash 校验通过');
  }
  function fail(e){playing=false;say(e.message);$('canvas-loading').hidden=false;$('canvas-loading').textContent=e.message;$('integrity').textContent='验证失败';}
  $('play').onclick=()=>{if(!player)return;if(player.sim.t===doc.actions.length)seek(0);playing=!playing;update();hashPosition();};
  $('replay-start').onclick=()=>{seek(0);playing=true;update();};$('step-back').onclick=()=>seek(Math.max(0,player.sim.t-1));$('step-next').onclick=()=>seek(Math.min(doc.actions.length,player.sim.t+1));$('seek').oninput=()=>seek(Number($('seek').value));$('filter').oninput=()=>list();$('zoom').oninput=()=>timeline();
  $('timeline').onclick=e=>{const rect=$('timeline').getBoundingClientRect();seek(Math.round(Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*doc.actions.length));};
  $('reward-chart').onclick=e=>{const rect=e.target.getBoundingClientRect();seek(Math.round((e.clientX-rect.left)/rect.width*doc.actions.length));};
  $('curve-component').onchange=chart;$('curve-mode').onchange=chart;$('reward-player').onchange=()=>update();
  $('advanced-close').addEventListener('click',pause);$('advanced-forget').addEventListener('click',pause);
  async function copy(value){try{await navigator.clipboard.writeText(value);say('已复制');}catch(_){$('locator').focus();$('locator').select();say('剪贴板不可用；定位已选中');}}
  $('copy-position').onclick=()=>copy(P.locator(doc,player.sim.t));$('copy-link').onclick=()=>{hashPosition();copy(location.href);};
  $('add-annotation').onclick=()=>form();$('cancel-annotation').onclick=()=>{$('annotation-form').hidden=true;};
  $('annotation-form').onsubmit=e=>{e.preventDefault();const a={id:editing||crypto.randomUUID(),tick:Number($('annotation-tick').value),category:$('annotation-category').value,severity:$('annotation-severity').value,note:$('annotation-note').value};const next=rows.filter(x=>x.id!==a.id).concat(a);try{P.validateAnnotations(P.annotationDocument(doc,next),doc);rows=next;persist();$('annotation-form').hidden=true;}catch(e){say(e.message);}};
  $('export-annotations').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(P.annotationDocument(doc,rows),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('import-annotations').onclick=()=>$('annotation-file').click();$('annotation-file').onchange=async()=>{try{const file=$('annotation-file').files[0];if(!file)return;if(file.size>1024*1024)throw Error('标注文件过大');const next=P.validateAnnotations(JSON.parse(await file.text()),doc);rows=next;persist();say('标注导入成功');}catch(e){say(`导入失败：${e.message}`);}finally{$('annotation-file').value='';}};
  window.addEventListener('hashchange',()=>{const p=new URLSearchParams(location.hash.slice(1)),id=p.get('replay'),tick=Number(p.get('tick')||0);if(!Number.isInteger(tick)||tick<0){say('非法深链 tick');return;}if(doc?.meta.id===id)seek(Math.min(tick,doc.actions.length));else load(id,tick).catch(fail);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  function frame(now){const dt=Math.min(now-last,250);last=now;if(playing&&player&&!$('advanced-settings').hidden){accumulator+=dt*Number($('speed').value);let moved=false;while(accumulator>=100&&playing){accumulator-=100;if(player.step()){renderer.addExplosion(player.info,player.sim.t*100);moved=true;}if(player.sim.t===doc.actions.length)playing=false;}if(moved){update();hashPosition();}}draw();requestAnimationFrame(frame);}requestAnimationFrame(frame);
  try{const r=await fetch('review-catalog.json');if(!r.ok)throw Error(`目录 HTTP ${r.status}`);catalog=await r.json();if(catalog.schema!=='qqt.diagnostic-catalog/v1')throw Error('目录 schema');$('count').textContent=catalog.replays.length;$('catalog-meta').textContent=`${catalog.replays.length} 局 · H7 g32\n${catalog.screened_episodes} 个固定 seed 已筛查\n${catalog.raw_bytes} bytes / gzip ${catalog.gzip_bytes} bytes`;list();const p=new URLSearchParams(location.hash.slice(1));await load(p.get('replay')||catalog.replays[0].id,Number(p.get('tick')||0));}catch(e){fail(e);}
})();
