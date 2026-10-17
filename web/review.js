'use strict';
(async function() {
  const P=QQTReview,V=QQTReviewView,D=QQTReviewDagger,A=QQTReviewAdvantage,$=id=>document.getElementById(id);
  const auth=QQTDeviceAuth.create(localStorage,crypto);
  QQTPanels.mount(document,{auth});
  if(auth.authorized())$('advanced-open').click();
  const labels={enemy_drop_unpicked:'掉包后未拾',window_closed_empty:'窗口未切换',return_deviation:'返程偏航',cage_no_delivery:'到笼未交付',self_bomb:'自炸',stagnation:'停滞',bomb:'放泡',use_item:'使用道具',death:'死亡',revive:'复活',control:'进入糖泡',control_end:'解除控制',drop:'掉包',native_transfer:'接触转移',pickup:'拾取',stock_theft:'偷包',delivery:'交付',phase:'阶段切换'};
  let catalog,doc,player,preview,renderer,playing=false,accumulator=0,last=performance.now(),rows=[],editing=null,loadToken=0;
  let lastDraw=0,frameStart=performance.now(),frameCount=0,actualFPS=0,rewardByTick=new Map(),curveCache=null;
  let daggerCatalog,daggerLoadToken=0;
  const daggerCache=new Map();
  const text=(tag,value,className)=>{const e=document.createElement(tag);e.textContent=value;if(className)e.className=className;return e;};
  const say=value=>$('notice').textContent=value;
  const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
  const number=n=>Math.abs(n)<1e-8?'0':n.toFixed(3);
  const historyControl=text('label','','history-control'),historyToggle=document.createElement('input');historyToggle.type='checkbox';historyToggle.id='show-history';historyControl.append(historyToggle,document.createTextNode('历史 H7 录像'));$('filter').after(historyControl);historyToggle.onchange=list;
  const advantagePanel=document.createElement('section');
  advantagePanel.innerHTML='<h3>模型动作 · 一步优势</h3><p id="advantage-status"></p><div id="advantage-details"></div>';
  $('state-details').before(advantagePanel);
  const currentDagger=document.createElement('section');
  currentDagger.innerHTML='<h3>当前 tick · DAgger</h3><p id="dagger-current-status"></p><div id="dagger-current"></div>';
  $('teacher').previousElementSibling.before(currentDagger);
  function metricList(target,values){
    const dl=document.createElement('dl');dl.className='dagger-metrics';
    for(const [key,v]of values){const row=text('div',''),name=text('dt',D.FIELDS[key]||'未翻译');name.append(text('small',key));
      row.append(name,text('dd',D.value(v,['teacher_acceptance','supervision_coverage','legal_action_rate','counterfactual_survival','task_progress','drop_expert_consistency','return_expert_consistency'].includes(key))));dl.append(row);}
    target.replaceChildren(dl);
  }
  function daggerTick(){
    $('dagger-current-status').textContent=D.episodeStatus(doc);$('dagger-current').replaceChildren();
    const records=D.atTick(doc,player.sim.t);
    if(!records.length){$('dagger-current').append(text('p','当前 tick：未记录'));return;}
    for(const r of records){const box=text('div','');box.append(text('strong',`${r.player===0?'模型 P0':'对手 P1'} · ${r.fragment?D.label(r.fragment):'片段类别：未记录'}`));
      for(const key of ['candidate','explicit_localization','teacher_proposed','strict_accepted','actor_query','actor_gradient'])box.append(text('div',`${D.label(key)}：${D.value(r[key])}`));
      box.append(text('div',`拒标原因 · rejection_reasons：${r.rejection_reasons==null?'未记录':r.rejection_reasons.length?r.rejection_reasons.map(D.label).join('；'):'无'}`),
        text('div',`动作集合 · action_set [move, ability]：${r.action_set==null?'未记录':r.action_set.map(a=>'['+a.join(',')+']').join(' / ')}`));$('dagger-current').append(box);}
  }
  function daggerEpisode(){
    $('dagger-episode-status').textContent=D.episodeStatus(doc);
    const related=daggerCatalog?.runs.find(r=>r.checkpoint_sha256===doc.meta.checkpoint_sha256);
    $('dagger-association').textContent=related?`同 checkpoint 的 ${related.title} · run 级数据，不是本局采集`:'同 checkpoint / run 的质量汇总：未记录';
    const counts={};for(const e of doc.dagger_events||[])if(e.fragment)counts[e.fragment]=(counts[e.fragment]||0)+1;
    $('dagger-episode-counts').replaceChildren(text('p',Object.entries(D.episodeCounts(doc)).map(([k,v])=>
      `${D.label(k)}：${v?`${v.positive} / ${v.recorded} 已记录`:'未记录'}`).join('；')),
      text('p',Object.entries(counts).map(([k,v])=>`${D.label(k)} ${v} 条`).join('；')||'片段类别 / 动作集合：未记录'));
    daggerTick();
  }
  async function daggerSummary(id){
    const request=++daggerLoadToken,row=daggerCatalog.runs.find(r=>r.id===id);if(!row)throw Error('未知 DAgger 汇总');
    if(!daggerCache.has(id)){const response=await fetch('review-dagger/'+row.file);if(!response.ok)throw Error('DAgger artifact HTTP');
      const bytes=new Uint8Array(await response.arrayBuffer());if(bytes.length!==row.bytes||await P.sha256(bytes)!==row.sha256)throw Error('DAgger artifact hash');
      daggerCache.set(id,D.validateQuality(JSON.parse(new TextDecoder().decode(bytes))));}
    if(request!==daggerLoadToken)return;const q=daggerCache.get(id);
    const related=q.checkpoint_sha256===doc.meta.checkpoint_sha256;
    const generations=q.generations.length>1?`g${q.generations[0]} - g${q.generations.at(-1)}（${q.generations.length} 代）`:`g${q.generations[0]}`;
    $('dagger-scope').textContent=`${related?'同模型 checkpoint 的 run 汇总':q.scope==='smoke'?'独立 Smoke，非本局 / 非本 checkpoint':'历史参考，非本局 / 非本 checkpoint'} · scope=${q.scope} · ${generations} · ${q.source_scopes.join('+')}。本局逐tick 记录与汇总分别采集。`;
    $('dagger-validation').textContent=q.validation?'严格验收失败 · FAILED：'+q.validation.failed_requirements.map(D.label).join('；'):
      q.accepted_is_query_lower_bound?'历史计数口径：accepted 是 query 下界；定位未采集。':'质量快照；未采集的定位、拒标细因与反事实指标保持未记录。';
    metricList($('dagger-funnel'),Object.entries(q.funnel));$('dagger-smoke').replaceChildren();
    if(q.smoke_a){metricList($('dagger-smoke'),Object.entries(q.smoke_a));$('dagger-smoke').append(text('p',`己包回收不能替代敌源拾取。原 artifact passed=${q.validation.original_passed}；strict_declared_acceptance=${q.validation.strict_declared_acceptance}，严格链路验收失败。`));}
    const table=$('dagger-fragments');table.replaceChildren();const header=document.createElement('tr');
    for(const title of ['片段类别 / machine key','候选','教师提议','严格接受','Actor 梯度查询'])header.append(text('th',title));const head=document.createElement('thead');head.append(header);table.append(head);
    const body=document.createElement('tbody');for(const [fragment,values]of Object.entries(q.fragments)){const tr=document.createElement('tr'),name=text('td',D.FRAGMENTS[fragment]||'未翻译');name.append(text('small',fragment));tr.append(name);for(const key of ['candidate','teacher_proposed','strict_accepted','actor_query'])tr.append(text('td',D.value(values[key])));body.append(tr);}table.append(body);
    $('dagger-quality').replaceChildren();for(const values of [q.teacher_quality,q.refusal_reasons]){const section=text('div','');metricList(section,Object.entries(values));$('dagger-quality').append(section);}
    $('dagger-quality').append(text('p',`教师动作集合大小 · action_set_size_histogram：${q.action_set_size_histogram?Object.entries(q.action_set_size_histogram).map(([k,v])=>`${k} 个动作 = ${D.value(v)}`).join('；'):'未记录'}。动作集合的逐动作内容：未记录。`));
    $('dagger-sources').replaceChildren(text('p',`source run: ${q.source_run}`),text('p',`来源生成时间: ${q.source_generated_at||'未记录'} · 转换时间: ${q.generated_at}`),text('p',`artifact SHA-256: ${row.sha256}`));
    for(const source of q.sources){const link=text('a',source.file);link.href='review-dagger/sources/'+source.file;link.target='_blank';link.rel='noopener';const line=text('div',`schema: ${source.schema||'未记录'} · source SHA-256: ${source.sha256} · ${source.path} `);line.append(link);if(source.original_sha256)line.append(text('div',`原始日志 SHA-256: ${source.original_sha256}`));$('dagger-sources').append(line);}
  }
  function hashPosition(){if(doc)history.replaceState(null,'',`#replay=${encodeURIComponent(doc.meta.id)}&tick=${player.sim.t}`);}
  function pause(){playing=false;accumulator=0;update();draw();}
  function preparePreview(){preview.seek(Math.min(player.sim.t+1,doc.actions.length));}
  function start(){if(!player)return;if(player.sim.t===doc.actions.length)seek(0);last=performance.now();frameStart=last;frameCount=0;lastDraw=last-1000/Number($('frame-rate').value);playing=true;update();}
  function seek(tick){if(!player)return;playing=false;accumulator=0;renderer.reset();player.seek(tick);preparePreview();const info=player.info;if(info)renderer.addExplosion(info,tick*100);update();hashPosition();draw();}
  function list(){
    const filter=$('filter').value.toLowerCase();$('replay-list').replaceChildren();
    const visible=[...catalog.replays,...(historyToggle.checked?catalog.history_replays||[]:[])];
    for(const r of visible){
      if(!`${r.id} ${r.summary.failures.map(k=>labels[k]).join(' ')}`.toLowerCase().includes(filter))continue;
      const b=text('button','',`replay-row${doc?.meta.id===r.id?' active':''}`);b.dataset.replay=r.id;
      b.append(text('strong',`${r.model} · ${r.variant==='open'?'清砖图':'原始图'}`),text('small',`seed ${r.seed} · ${P.time(r.summary.ticks)} · ${r.summary.winner===null?'平局':r.summary.winner===0?'模型胜':'对手胜'}`));
      const tags=text('div','','tags');for(const k of r.summary.failures)tags.append(text('span',labels[k]||k,'tag'));b.append(tags);b.onclick=()=>load(r.id,0).catch(fail);$('replay-list').append(b);
    }
  }
  function details(target,values){const dl=document.createElement('dl');for(const [key,value] of values){const div=document.createElement('div');div.append(text('dt',key),text('dd',value));dl.append(div);}target.replaceChildren(dl);}
  function currentReward(tick){return rewardByTick.get(tick)?.[Number($('reward-player').value)+1]||{};}
  function rewardValues(){const all={};for(const row of doc.rewards){if(row[0]>player.sim.t)break;for(const [key,v] of Object.entries(row[Number($('reward-player').value)+1]))all[key]=(all[key]||0)+v;}return all;}
  function update(){
    if(!player)return;const s=player.sim,t=s.t;
    $('exact-time').textContent=`tick ${t} · ${P.time(t)}`;$('play-state').textContent=playing?'播放中':'已暂停';$('position-panel').classList.toggle('playing',playing);
    $('locator').value=P.locator(doc,t);$('seek').value=t;$('playhead').style.left=`${t/doc.actions.length*100}%`;
    $('play').title=playing?'暂停':'播放';$('play').setAttribute('aria-label',playing?'暂停':'播放');$('play-icon').src=`assets/review-icons/${playing?'pause':'play'}.svg`;
    const delta=currentReward(t),total=rewardValues();$('reward-delta').textContent=number(sum(delta));$('reward-total').textContent=number(sum(total));
    $('components').replaceChildren();for(const k of componentNames()){const translated=V.reward(k),e=text('div','',`component${translated.known?'':' untranslated'}`),v=delta[k]||0,name=text('span',translated.name);e.title=translated.help;e.dataset.component=k;if($('english-rewards').checked)name.append(text('small',k));e.append(name,text('span',`${number(v)} / ${number(total[k]||0)}`,v<0?'negative':'positive'));$('components').append(e);}
    const a=t?doc.actions[t-1]:null;
    const advantage=A.atTick(doc,t),value=v=>v===null?'未记录':number(v);
    details($('advantage-details'),[['选中模型动作 [move, ability]',a?`${a[0]}, ${a[1]}`:'开局'],['执行动作 Q₁',value(advantage.taken_q)],['策略平均 Q₁',value(advantage.mean_q)],['策略加权一步优势 A₁',value(advantage.advantage)],['优势增量 ΔA₁',value(advantage.delta)]]);
    $('advantage-status').textContent=doc.meta.advantage?.status==='recorded'?`联合动作 · γ=${doc.meta.advantage.gamma}`:`未记录：${doc.meta.advantage?.reason||'历史录像未采集优势'}`;
    details($('state-details'),[['模型阶段',P.phase(s)],['对手阶段',P.phase(s,1)],['动作 [move, ability]',a?`${a[0]},${a[1]} | ${a[2]},${a[3]}`:'开局'],['包归属（模型 / 对手）',s.bunCarried.map(v=>v<0?'空手':`源队 ${v}`).join(' / ')],['糖泡剩余 tick',s.trapped.join(' / ')],['死亡 / 复活倒计时',s.bunRespawn.join(' / ')],['坐标 [row,col]',`${s.pos[0].toFixed(2)},${s.pos[1].toFixed(2)} | ${s.pos[2].toFixed(2)},${s.pos[3].toFixed(2)}`],['笼内存包',s.bunStored.map(a=>a.join(',')).join(' / ')],['终局',s.done?`胜者 ${s.winner===null?'平局':s.winner}`:'进行中']]);
    $('current-events').replaceChildren();const events=doc.events.filter(e=>e.tick===t);
    if(events.length)$('current-events').append(text('p',rewardEvidence(t),'event-reward'));
    for(const e of events)$('current-events').append(text('div',`${labels[e.kind]||e.kind} · ${e.player===0?'模型 P0':'对手 P1'}${e.origin===undefined?'':` · 源队${e.origin}`}${e.to?` → ${e.to}`:''}`,'event-line'));if(!events.length)$('current-events').append(text('span','—','empty'));
    chart();
    daggerTick();
  }
  function componentNames(){return V.componentNames(doc);}
  function curveOptions(){const selected=$('curve-component').value;$('curve-component').replaceChildren(new Option('总奖励','total'));for(const k of componentNames()){const r=V.reward(k),option=new Option(r.name+($('english-rewards').checked?` (${k})`:''),k);option.title=r.help;$('curve-component').add(option);}if([...$('curve-component').options].some(o=>o.value===selected))$('curve-component').value=selected;}
  function chart(){
    if(!doc)return;const c=$('reward-chart'),g=c.getContext('2d'),w=c.width,h=c.height,component=$('curve-component').value,mode=$('curve-mode').value;
    const key=`${doc.meta.id}:${$('reward-player').value}:${component}:${mode}`;
    if(curveCache?.key!==key){
      const base=document.createElement('canvas');base.width=w;base.height=h;const bg=base.getContext('2d');
      const values=new Array(doc.actions.length+1).fill(0);let total=0;for(let t=0;t<values.length;t++){const r=currentReward(t),v=component==='total'?sum(r):r[component]||0;total+=v;values[t]=mode==='cumulative'?total:v;}
      const lo=Math.min(0,...values),hi=Math.max(1,...values),y=v=>h-15-(v-lo)/(hi-lo)*(h-30);bg.strokeStyle='#343b46';bg.beginPath();bg.moveTo(0,y(0));bg.lineTo(w,y(0));bg.stroke();bg.fillStyle='#8492a5';bg.font='18px monospace';bg.fillText(hi.toFixed(1),5,18);bg.fillText(lo.toFixed(1),5,h-3);bg.strokeStyle='#74cdb2';bg.lineWidth=2;bg.beginPath();values.forEach((v,t)=>t?bg.lineTo(t/(values.length-1)*w,y(v)):bg.moveTo(0,y(v)));bg.stroke();curveCache={key,base};
    }
    g.clearRect(0,0,w,h);g.drawImage(curveCache.base,0,0);g.strokeStyle='#e9f7ef';const x=player.sim.t/doc.actions.length*w;g.beginPath();g.moveTo(x,0);g.lineTo(x,h);g.stroke();
  }
  function rewardEvidence(tick){const r=currentReward(tick);return `${$('reward-player').value==='0'?'模型':'对手'} Reward ${number(sum(r))}（同 tick 事件共用一次） · ${Object.entries(r).map(([k,v])=>`${V.reward(k).name} ${number(v)}`).join('；')||'无非零分量'}`;}
  function marker(track,tick,color,title,width){const b=text('button','','mark'+(width?' phase-mark':''));b.dataset.tick=tick;b.style.left=`${tick/doc.actions.length*100}%`;b.style.background=color;b.title=`tick ${tick} · ${P.time(tick)} · ${title}`;b.setAttribute('aria-label',b.title);if(width)b.style.width=`${width/doc.actions.length*100}%`;b.onclick=e=>{e.stopPropagation();seek(tick);};$(track).append(b);return b;}
  function numericTrack(track,entries,format){
    const lanes=[],width=$('timeline-scroll').clientWidth*Number($('zoom').value);
    for(const entry of entries){const x=entry.tick/doc.actions.length*width;
      const b=marker(track,entry.tick,entry.value<0?'#603b43':'#254c41',entry.title);b.textContent=format(entry.value);b.classList.add('numeric-mark');
      let lane=lanes.findIndex(end=>x>=end+4);if(lane<0)lane=lanes.length;lanes[lane]=x+b.offsetWidth;
      b.style.top=`${7+lane*22}px`;b.dataset.value=entry.value;
    }
    const height=Math.max(36,lanes.length*22+14);$(track).style.height=height+'px';document.querySelector(`[data-track-label="${track}"]`).style.height=height+'px';
  }
  function timeline(){
    for(const id of ['ruler','phase-track','event-track','reward-track','advantage-track','advantage-delta-track','dagger-track','annotation-track'])$(id).replaceChildren();
    $('timeline').style.width=`${Number($('zoom').value)*100}%`;
    for(let t=0;t<=doc.actions.length;t+=200){const e=text('span',P.time(t).slice(0,5));e.style.left=`${t/doc.actions.length*100}%`;$('ruler').append(e);}
    let start=0,phase='战斗';for(const e of doc.events.filter(e=>e.kind==='phase'&&e.player===0)){marker('phase-track',start,phase==='持包返程'?'#77c6aa':phase==='运输窗口'?'#d4ab64':'#5ba6ba',phase,e.tick-start);start=e.tick;phase=e.to;}marker('phase-track',start,'#5ba6ba',phase,doc.actions.length-start);
    const eventGroups=new Map();for(const e of doc.events.filter(e=>e.kind!=='phase'&&e.kind!=='bomb')){if(!eventGroups.has(e.tick))eventGroups.set(e.tick,[]);eventGroups.get(e.tick).push(e);}
    numericTrack('event-track',[...eventGroups].sort(([a],[b])=>a-b).map(([tick,events])=>({tick,value:sum(currentReward(tick)),title:events.map(e=>`${e.player===0?'模型':'对手'} ${labels[e.kind]||e.kind}`).join('；')+' · '+rewardEvidence(tick)})),v=>'R '+(v>0?'+':'')+number(v));
    const rewardPid=Number($('reward-player').value);for(const r of doc.rewards){const components=r[rewardPid+1];if(Object.keys(components).length)marker('reward-track',r[0],sum(components)<0?'#e78f94':'#7bcaac',`${rewardPid===0?'模型':'对手'}奖励增量 ${number(sum(components))} · ${Object.entries(components).map(([k,v])=>`${V.reward(k).name} ${number(v)}`).join('；')}`);}
    for(const a of rows)marker('annotation-track',a.tick,'#e192a6',a.note||a.category);
    D.renderTrack($('dagger-track'),doc,seek,P.time);
    for(const [track,key]of [['advantage-track','advantage'],['advantage-delta-track','delta']]){
      const entries=(doc.advantages?.ticks||[]).map(tick=>({tick,value:A.atTick(doc,tick)[key]})).filter(e=>e.value!==null).map(e=>({...e,title:`${key==='delta'?'优势增量':'策略加权一步优势'} ${number(e.value)} · 联合动作`}));
      numericTrack(track,entries,v=>(v>0?'+':'')+number(v));if(!entries.length)$(track).append(text('span','未记录','dagger-empty'));
    }
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
  function draw(){if(renderer&&player){const nextAction=doc.actions[player.sim.t],intents=nextAction?[nextAction[0],nextAction[2]]:[4,4];const motion=V.motion(player.sim,preview.sim,playing?accumulator/100:0,playing&&$('smooth').checked,doc.events,intents);motion.intents=intents;renderer.render(player.sim,player.sim.t*100+motion.fraction*100,motion);$('refresh-status').textContent=`实际 ${actualFPS} FPS · 模拟 10 tick/s · α ${motion.fraction.toFixed(3)} · 目标 ${$('frame-rate').value} FPS`;}}
  async function load(id,tick){
    const token=++loadToken;playing=false;$('canvas-loading').hidden=false;$('canvas-loading').textContent='正在验证录像与状态 hash';$('integrity').textContent='验证中';
    const row=[...catalog.replays,...(catalog.history_replays||[])].find(r=>r.id===id);if(!row)throw Error('未知 replay ID');
    const response=await fetch(`review-replays/${row.file}`);if(!response.ok)throw Error(`录像 HTTP ${response.status}`);const bytes=new Uint8Array(await response.arrayBuffer());if(bytes.length!==row.bytes||await P.sha256(bytes)!==row.sha256)throw Error('录像文件 hash / 大小不匹配');
    const next=P.validate(JSON.parse(new TextDecoder().decode(bytes)));const mapResponse=await fetch('assets/maps/levels.json');if(!mapResponse.ok)throw Error('地图 HTTP');const level=(await mapResponse.json()).find(x=>x.qqt_id===806);if(await P.hash(level)!==next.meta.map_sha256)throw Error('地图 hash');
    const simBytes=await(await fetch('sim.js')).arrayBuffer();if(await P.sha256(simBytes)!==next.meta.simulator_sha256)throw Error('模拟器版本不匹配');
    const nextPlayer=new P.Player(next,level);await nextPlayer.verify();const assets=await QQTVisual.loadAssets(level);if(token!==loadToken)return;
    Object.defineProperty(next,'asset_sha256',{value:row.sha256});
    doc=next;player=nextPlayer;preview=new P.Player(doc,level);preview.cache=player.cache;rewardByTick=new Map(doc.rewards.map(r=>[r[0],r]));curveCache=null;renderer=QQTVisual.createRenderer($('game'),level,assets);rows=[];editing=null;$('annotation-form').hidden=true;
    try{const saved=localStorage.getItem(storageKey());if(saved)rows=P.validateAnnotations(JSON.parse(saved),doc);}catch(_){say('已有标注无效或存储不可用');}
    $('episode-title').textContent=`模型 P0 · ${doc.meta.model} 对阵 对手 P1`;$('episode-meta').textContent=`对手 ${doc.meta.opponent} · ${doc.meta.variant==='open'?'806 清砖图':'806 原始图'} · seed ${doc.meta.seed} · checkpoint ${doc.meta.checkpoint_sha256.slice(0,12)}`;$('terminal').textContent=`终局：${doc.summary.winner===null?'平局':doc.summary.winner===0?'模型胜':'对手胜'}`;
    $('seek').max=doc.actions.length;$('duration').textContent=P.time(doc.actions.length);$('integrity').style.color='#82d2b6';
    curveOptions();daggerEpisode();
    if(daggerCatalog){const related=daggerCatalog.runs.find(r=>r.checkpoint_sha256===doc.meta.checkpoint_sha256);if(related)$('dagger-run').value=related.id;await daggerSummary($('dagger-run').value);}
    if(token!==loadToken)return;
    details($('teacher'),['eligible','query','accept','reject'].map(k=>[k,doc.meta.teacher?.[k]??'未测量']));
    $('provenance').replaceChildren();for(const [k,v] of [['Episode',doc.meta.episode_id],['Checkpoint SHA',doc.meta.checkpoint_sha256],['Model SHA',doc.meta.model_sha256],['Simulator SHA',doc.meta.simulator_sha256],['Opponent SHA',doc.meta.opponent_sha256],['Replay SHA',row.sha256],['Reward',doc.meta.reward_source||'未测量'],['原始 / gzip',`${row.bytes} / ${row.gzip_bytes} bytes`]])$('provenance').append(text('div',`${k}: ${v}`));
    $('canvas-loading').hidden=true;list();timeline();annotations();seek(Math.min(doc.actions.length,Math.max(0,Number.isInteger(tick)?tick:0)));$('integrity').textContent='hash 已验证';say('录像与间隔状态 hash 校验通过');
  }
  function fail(e){playing=false;say(e.message);$('canvas-loading').hidden=false;$('canvas-loading').textContent=e.message;$('integrity').textContent='验证失败';}
  $('play').onclick=()=>{if(!player)return;if(playing)pause();else start();hashPosition();};
  $('replay-start').onclick=()=>{seek(0);start();};$('step-back').onclick=()=>seek(Math.max(0,player.sim.t-1));$('step-next').onclick=()=>seek(Math.min(doc.actions.length,player.sim.t+1));$('seek').oninput=()=>seek(Number($('seek').value));$('filter').oninput=()=>list();$('zoom').oninput=()=>timeline();
  $('timeline').onclick=e=>{const rect=$('timeline').getBoundingClientRect();seek(Math.round(Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*doc.actions.length));};
  $('reward-chart').onclick=e=>{const rect=e.target.getBoundingClientRect();seek(Math.round((e.clientX-rect.left)/rect.width*doc.actions.length));};
  $('curve-component').onchange=chart;$('curve-mode').onchange=chart;$('reward-player').onchange=()=>{update();timeline();};$('english-rewards').onchange=()=>{curveOptions();update();};$('smooth').onchange=draw;
  $('dagger-run').onchange=()=>daggerSummary($('dagger-run').value).catch(e=>say(e.message));
  $('frame-rate').onchange=()=>{lastDraw=0;frameStart=performance.now();frameCount=0;$('refresh-status').textContent=`目标 ${$('frame-rate').value} fps · 逻辑 10 tick/s`;};
  $('advanced-close').addEventListener('click',pause);$('advanced-forget').addEventListener('click',pause);
  async function copy(value){try{await navigator.clipboard.writeText(value);say('已复制');}catch(_){$('locator').focus();$('locator').select();say('剪贴板不可用；定位已选中');}}
  $('copy-position').onclick=()=>copy(P.locator(doc,player.sim.t));$('copy-link').onclick=()=>{hashPosition();copy(location.href);};
  $('add-annotation').onclick=()=>form();$('cancel-annotation').onclick=()=>{$('annotation-form').hidden=true;};
  $('annotation-form').onsubmit=e=>{e.preventDefault();const a={id:editing||crypto.randomUUID(),tick:Number($('annotation-tick').value),category:$('annotation-category').value,severity:$('annotation-severity').value,note:$('annotation-note').value};const next=rows.filter(x=>x.id!==a.id).concat(a);try{P.validateAnnotations(P.annotationDocument(doc,next),doc);rows=next;persist();$('annotation-form').hidden=true;}catch(e){say(e.message);}};
  $('export-annotations').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(P.annotationDocument(doc,rows),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('import-annotations').onclick=()=>$('annotation-file').click();$('annotation-file').onchange=async()=>{try{const file=$('annotation-file').files[0];if(!file)return;if(file.size>1024*1024)throw Error('标注文件过大');const next=P.validateAnnotations(JSON.parse(await file.text()),doc);rows=next;persist();say('标注导入成功');}catch(e){say(`导入失败：${e.message}`);}finally{$('annotation-file').value='';}};
  window.addEventListener('hashchange',()=>{const p=new URLSearchParams(location.hash.slice(1)),id=p.get('replay'),tick=Number(p.get('tick')||0);if(!Number.isInteger(tick)||tick<0){say('非法深链 tick');return;}if(doc?.meta.id===id)seek(Math.min(tick,doc.actions.length));else load(id,tick).catch(fail);});
  window.addEventListener('resize',()=>{if(doc)timeline();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  function frame(now){
    const dt=Math.min(now-last,250);last=now;
    if(playing&&player&&!$('advanced-settings').hidden){
      accumulator+=dt*Number($('speed').value);let moved=false;
      while(accumulator>=100&&playing){accumulator-=100;if(player.step()){renderer.addExplosion(player.info,player.sim.t*100);moved=true;}if(player.sim.t===doc.actions.length){playing=false;accumulator=0;}}
      if(moved){preparePreview();update();hashPosition();}
      const interval=1000/Number($('frame-rate').value),elapsed=now-lastDraw;
      if(elapsed>=interval-.5){draw();lastDraw=now-(Math.max(0,elapsed)%interval);frameCount++;}
      if(now-frameStart>=1000){actualFPS=Math.round(frameCount*1000/(now-frameStart));frameStart=now;frameCount=0;}
    }
    requestAnimationFrame(frame);
  }requestAnimationFrame(frame);
  try{const r=await fetch('review-catalog.json');if(!r.ok)throw Error(`目录 HTTP ${r.status}`);catalog=await r.json();if(catalog.schema!=='qqt.diagnostic-catalog/v1')throw Error('目录 schema');
    const dr=await fetch('review-dagger/catalog.json');if(!dr.ok)throw Error('DAgger 目录 HTTP');daggerCatalog=await dr.json();if(daggerCatalog.schema!=='qqt.dagger-catalog/v1')throw Error('DAgger 目录 schema');
    for(const row of daggerCatalog.runs)$('dagger-run').add(new Option(row.title+' · '+row.scope,row.id));
    $('count').textContent=catalog.replays.length;$('catalog-meta').textContent=`${catalog.replays.length} 局 · ${catalog.model}\n${catalog.screened_episodes} 个固定 seed 已筛查\n${catalog.raw_bytes} bytes / gzip ${catalog.gzip_bytes} bytes`;list();const p=new URLSearchParams(location.hash.slice(1));await load(p.get('replay')||catalog.replays[0].id,Number(p.get('tick')||0));}catch(e){fail(e);}
})();
