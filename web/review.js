'use strict';
(async function() {
  const P=QQTReview,V=QQTReviewView,D=QQTReviewDagger,A=QQTReviewAdvantage,R=QQTReviewRoute,F=QQTReviewReturn,M=QQTReviewMetrics,$=id=>document.getElementById(id);
  const auth=QQTDeviceAuth.create(localStorage,crypto);
  QQTPanels.mount(document,{auth});
  if(auth.authorized())$('advanced-open').click();
  const labels={enemy_drop_unpicked:'掉包后未拾',window_closed_empty:'窗口未切换',return_deviation:'返程偏航',cage_no_delivery:'到笼未交付',self_bomb:'自炸',stagnation:'停滞',approx_phi_discontinuity:'近似势能跨步不连续',match_loss:'对局失利',no_delivery:'未交付',bomb:'放泡',use_item:'使用道具',death:'死亡',revive:'复活',control:'进入糖泡',control_end:'解除控制',drop:'掉包',native_transfer:'接触转移',pickup:'拾取',stock_theft:'偷包',delivery:'交付',phase:'阶段切换'};
  let catalog,doc,player,preview,renderer,playing=false,accumulator=0,last=performance.now(),rows=[],editing=null,loadToken=0;
  let lastDraw=0,frameStart=performance.now(),frameCount=0,actualFPS=0,rewardByTick=new Map(),curveCache=null;
  let daggerCatalog,daggerLoadToken=0;
  const daggerCache=new Map();
  let returnAnalysis={points:[],intervals:[]},returnAudit=null;
  const text=(tag,value,className)=>{const e=document.createElement(tag);e.textContent=value;if(className)e.className=className;return e;};
  const say=value=>$('notice').textContent=value;
  const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
  const number=n=>n===0?'0':Math.abs(n)<.001?Number(n.toPrecision(5)).toString():n.toFixed(3);
  const historyControl=text('label','','history-control'),historyToggle=document.createElement('input');historyToggle.type='checkbox';historyToggle.id='show-history';historyControl.append(historyToggle,document.createTextNode('历史 H7 / H8 录像'));$('filter').after(historyControl);historyToggle.onchange=list;
  const sceneFilter=document.createElement('select');sceneFilter.id='scene-filter';sceneFilter.setAttribute('aria-label','场景筛选');sceneFilter.add(new Option('全部场景',''));historyControl.before(sceneFilter);sceneFilter.onchange=list;
  const modelFilter=document.createElement('select');modelFilter.id='model-filter';modelFilter.setAttribute('aria-label','模型与能力标签');modelFilter.add(new Option('全部模型 / 能力标签',''));sceneFilter.before(modelFilter);modelFilter.onchange=()=>{const available=catalog.replays.filter(r=>r.comparison_model_id===modelFilter.value);if(sceneFilter.value&&available.length&&!available.some(r=>r.scene===sceneFilter.value))sceneFilter.value='';list();const row=available.find(r=>!sceneFilter.value||r.scene===sceneFilter.value);if(row&&doc?.meta.comparison_model_id!==modelFilter.value)load(row.id,0).catch(fail);};
  const modelSummary=document.createElement('section');modelSummary.id='model-summary';modelSummary.hidden=true;$('state-details').before(modelSummary);
  const decisionPanel=document.createElement('section');decisionPanel.id='policy-decision';decisionPanel.innerHTML='<h3>模型实际策略 · 合法联合动作</h3><div id="policy-decision-details"></div>';$('state-details').after(decisionPanel);
  const advantagePanel=document.createElement('section');
  advantagePanel.innerHTML='<h3>原始任务 Critic · 一步优势</h3><p id="advantage-status"></p><div id="advantage-details"></div>';
  const shapedPanel=document.createElement('section');
  shapedPanel.innerHTML='<h3>塑形 Critic · 一步优势</h3><p id="shaped-status"></p><div id="shaped-details"></div>';
  const approxPanel=document.createElement('section');approxPanel.id='je10-advantage';approxPanel.hidden=true;approxPanel.innerHTML='<h3>raw A1 / shaped A1 / ΔA · 离线一步近似</h3><p id="je10-advantage-status"></p><div id="je10-advantage-details"></div>';
  const comparison=document.createElement('div');comparison.className='critic-comparison';comparison.append(advantagePanel,shapedPanel);
  const crossGap=document.createElement('p');crossGap.id='cross-critic-gap';
  const returnPanel=document.createElement('section');returnPanel.className='return-audit';returnPanel.innerHTML='<h3>返程行为诊断 · 观测与策略</h3><p id="return-status"></p><div id="return-details"></div><div class="dagger-table-scroll"><table id="return-candidates"></table></div><details><summary>优势为何未记录 · 两代 Critic 与训练链路</summary><div id="return-missing"></div></details>';
  const returnWarning=text('p','','return-warning');returnWarning.id='return-warning';returnWarning.hidden=true;document.querySelector('.seat-legend').after(returnWarning);
  const routePanel=document.createElement('section');routePanel.className='route-details-panel';
  routePanel.innerHTML='<h3>路线势能 · 人工辅助值</h3><p id="route-status"></p><p class="route-explanation">原始任务 Critic 估计任务回报；路线奖励是人工路线辅助值。塑形 Critic 只有经过塑形回报训练并验证才展示，不能用路线奖励代替。两种 Critic 的 A₁ 均是模型估计，不是观测收益。</p><div id="route-details"></div>';
  $('state-details').before(returnPanel,comparison,approxPanel,crossGap,routePanel);
  const experiment=document.createElement('section');experiment.id='route-experiment';experiment.hidden=true;experiment.className='route-experiment';
  experiment.innerHTML='<h3>路线 PBRS · g32 实验结论</h3><p>32代 / 67,108,864步；λ=0.05。诊断返程 pickup 16/16、25% 7/16、50% 6/16、home 6/16、capture 2/16。fixed32 g32 capture 主评 / holdout 均 0/16；full2400 pickup / capture 均 0/16。Tactical causal kill 12→9/64；Hunter hard 9→1/64。诊断返程有所进展，自然完整对局未改善，攻击效能下降，不建议当前 λ=.05 原样续训。</p><p id="diagnostic-warning">诊断场景，不代表自然完整对局成功</p><p>本局 trap 初态和掉包归因预置，非模型自然击杀；对手固定向上、不放泡。路线值来自诊断导出器，末 tick Phi 未归零，不能当作训练 PBRS 账本。原环境 Reward、训练总 Reward、原始 / 塑形 Critic A₁ 均未记录；同代塑形 Critic 参数存在，但缺同口径 reward 与联合反事实结果。</p><a href="review-evidence/route-v35-g32.json" target="_blank" rel="noopener">逐 tick 复算与来源 SHA-256</a>';
  document.querySelector('.display-controls').after(experiment);
  const currentDagger=document.createElement('section');
  new MutationObserver(()=>{if(doc&&!$('advanced-settings').hidden){timeline();draw();}}).observe($('advanced-settings'),{attributes:true,attributeFilter:['hidden']});
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
    let group=null;
    for(const r of visible){
      const model=catalog.model_comparison?.models.find(m=>m.checkpoint_sha256===r.checkpoint_sha256);
      if(modelFilter.value&&model?.id!==modelFilter.value)continue;
      if(sceneFilter.value&&r.scene!==sceneFilter.value)continue;
      const failures=Array.isArray(r.summary?.failures)?r.summary.failures:[];
      if(!`${r.id} ${r.scene_name||''} ${failures.map(k=>labels[k]||k).join(' ')} ${r.diagnostic_audit?'返程振荡 停滞':''}`.toLowerCase().includes(filter))continue;
      if(model&&group!==model.id){group=model.id;$('replay-list').append(text('h3',`${model.id} · ${model.label}`,'model-group-heading'));}
      const b=text('button','',`replay-row${doc?.meta.id===r.id?' active':''}`);b.dataset.replay=r.id;
      b.append(text('strong',`${r.model} · ${r.scene_name||(r.variant==='open'?'清砖图':'原始图')}`),text('small',`seed ${r.seed} · ${P.time(r.summary.ticks)} · ${r.summary.done===false?'片段未终局':r.summary.winner===null?'平局':r.summary.winner===0?'模型胜':'对手胜'}`));
      if(r.diagnostic)b.append(text('strong',r.task==='recovery_threat'?'预置未归因控制窗口 · 真实空手取包 / 恢复诊断':r.scene==='carry_home'?'预置持包诊断，不代表自然完整对局成功':'训练指标离线诊断 · 完整自然对局','diagnostic-warning'));
      if(r.diagnostic_audit)b.append(text('span','返程振荡 / 停滞失败已审计','tag'));
      if(r.scene)b.append(text('small',`▦ ${r.map||806} · ${r.scene_rules} · map ${r.map_sha256.slice(0,8)}`,'scene-summary'));
      const tags=text('div','','tags');if(r.scene)tags.append(text('span',({native:'▦ 原生',open:'□ 开放',carry_home:'↩ 返程'})[r.scene]||r.scene,'badge'));for(const k of failures)tags.append(text('span',labels[k]||k,'tag'));b.append(tags);b.onclick=()=>load(r.id,0).catch(fail);$('replay-list').append(b);
    }
  }
  function details(target,values){const dl=document.createElement('dl');for(const [key,value] of values){const div=document.createElement('div');div.append(text('dt',key),text('dd',value));dl.append(div);}target.replaceChildren(dl);}
  function currentReward(tick,pid=Number($('reward-player').value)){return rewardByTick.get(tick)?.[pid+1]||{};}
  function rewardValues(){const all={};for(const row of doc.rewards){if(row[0]>player.sim.t)break;for(const [key,v] of Object.entries(row[Number($('reward-player').value)+1]))all[key]=(all[key]||0)+v;}return all;}
  function update(){
    if(!player)return;const s=player.sim,t=s.t;
    $('exact-time').textContent=`tick ${t} · ${P.time(t)}`;$('play-state').textContent=playing?'播放中':'已暂停';$('position-panel').classList.toggle('playing',playing);
    $('locator').value=P.locator(doc,t);$('seek').value=t;$('playhead').style.left=`${t/doc.actions.length*100}%`;
    $('play').title=playing?'暂停':'播放';$('play').setAttribute('aria-label',playing?'暂停':'播放');$('play-icon').src=`assets/review-icons/${playing?'pause':'play'}.svg`;
    const delta=currentReward(t),total=rewardValues(),rewardMeasured=doc.meta.reward_status!=='unrecorded';$('reward-delta').textContent=rewardMeasured?number(sum(delta)):'未记录';$('reward-total').textContent=rewardMeasured?number(sum(total)):'未记录';
    $('reward-chart').hidden=!rewardMeasured;$('curve-component').disabled=!rewardMeasured;$('curve-mode').disabled=!rewardMeasured;
    $('components').replaceChildren();for(const k of componentNames()){const translated=V.reward(k),e=text('div','',`component${translated.known?'':' untranslated'}`),v=delta[k]||0,name=text('span',translated.name);e.title=translated.help;e.dataset.component=k;if($('english-rewards').checked)name.append(text('small',k));e.append(name,text('span',`${number(v)} / ${number(total[k]||0)}`,v<0?'negative':'positive'));$('components').append(e);}
    const a=t?doc.actions[t-1]:null;
    const dec=M.decision(doc,t);decisionPanel.hidden=!doc.decisions;
    if(dec){const move=['上（0）','下（1）','左（2）','右（3）','停（4）'],ability=['无技能（0）','放泡（1）','道具（2）'],legal=[];for(let m=0;m<5;m++)for(let b=0;b<3;b++)if((dec.move_mask&(1<<m))&&(dec.ability_mask&(1<<b)))legal.push(`${m},${b}`);details($('policy-decision-details'),[['move 概率 · 合法掩码后',dec.move_probability.map((p,i)=>`${move[i]} ${(p*100).toFixed(3)}%${dec.move_mask&(1<<i)?'':'（非法）'}`).join('；')],['ability 概率 · 合法掩码后',dec.ability_probability.map((p,i)=>`${ability[i]} ${(p*100).toFixed(3)}%${dec.ability_mask&(1<<i)?'':'（非法）'}`).join('；')],['合法联合动作 [move, ability]',legal.join(' / ')],['选择方式','真实Actor greedy；两头概率乘积在合法联合空间重归一；概率量化步长1e-6']]);}else $('policy-decision-details').textContent='开局 / 历史录像未采集策略概率';
    const advantage=A.atTick(doc,t),advantageBlocked=doc.meta.advantage_audit?.status==='blocked',value=v=>v===null?(advantageBlocked?'无法严格复算':'未记录'):number(v);
    const advantageFields=[['选中模型动作 [move, ability]',a?`${a[0]}, ${a[1]}`:'开局'],['执行动作 Q₁',value(advantage.taken_q)],['策略平均 Q₁',value(advantage.mean_q)],['策略加权一步优势 A₁',value(advantage.advantage)],['优势增量 ΔA₁',value(advantage.delta)],['合法联合动作数',String(A.legalCount(doc,t)??'未记录')]];
    if(advantageBlocked)advantageFields.push(['严格复算证据',`checkpoint ${doc.meta.advantage_audit.checkpoint_sha256} · state ${doc.meta.advantage_audit.training_state_sha256} · policy ${doc.meta.advantage_audit.policy_sha256} · simulator ${doc.meta.advantage_audit.simulator_sha256}`],['Critic 来源',doc.meta.advantage_audit.critic_status],['原始任务 A₁',doc.meta.advantage_audit.raw_task_a1_status],['塑形 A₁',doc.meta.advantage_audit.shaped_a1_status]);
    details($('advantage-details'),advantageFields);
    $('advantage-status').textContent=doc.metrics?`offline_approximate_one_step · 同checkpoint真实Critic · move × ability联合动作 · γ=${doc.meta.metrics.gamma} · raw侧沿用total-reward Critic，非独立原始Critic` :doc.meta.advantage?.status==='recorded'?`move × ability 联合动作（含道具） · γ=${doc.meta.advantage.gamma} · 合法策略概率重归一 · ${doc.meta.advantage.measured_count} tick 已测`:advantageBlocked?`无法严格复算：${doc.meta.advantage_audit.reason}`:`未记录：${doc.meta.advantage?.reason||'历史录像未采集优势'}`;
    routeTick();
    returnTick();
    details($('state-details'),[['模型阶段',P.phase(s)],['对手阶段',P.phase(s,1)],['动作 [move, ability]',a?`${a[0]},${a[1]} | ${a[2]},${a[3]}`:'开局'],['包归属（模型 / 对手）',s.bunCarried.map(v=>v<0?'空手':`源队 ${v}`).join(' / ')],['糖泡剩余 tick',s.trapped.join(' / ')],['死亡 / 复活倒计时',s.bunRespawn.join(' / ')],['坐标 [row,col]',`${s.pos[0].toFixed(2)},${s.pos[1].toFixed(2)} | ${s.pos[2].toFixed(2)},${s.pos[3].toFixed(2)}`],['笼内存包',s.bunStored.map(a=>a.join(',')).join(' / ')],['终局',s.done?`胜者 ${s.winner===null?'平局':s.winner}`:'进行中']]);
    const diagnostic=doc.diagnostic_ticks?.[t-1];
    approxPanel.hidden=!doc.meta.approximate_a1&&!doc.metrics;
    advantagePanel.querySelector('h3').textContent=doc.metrics?'原始任务 Reward · 同 Critic 一步近似':'原始任务 Critic · 一步优势';
    if(doc.metrics){
      const m=doc.meta.metrics,v=M.atTick(doc,t),modelDelta=currentReward(t,0),n=x=>x==null?'未记录':x.toFixed(6);
      $('je10-advantage-status').textContent=m.limitations+' 公式：'+m.definition;
      details($('je10-advantage-details'),[['raw A1（近似）',n(v.raw_a1)],['shaped A1（近似）',n(v.shaped_a1)],['ΔA = shaped A1 − raw A1（近似）',n(v.delta_a)],['raw A1 相邻 tick 增量',n(v.raw_a1_increment)],['shaped A1 相邻 tick 增量',n(v.shaped_a1_increment)],['ΔA 相邻 tick 增量',n(v.delta_a_increment)],['raw Q1 执行动作 / 策略均值',n(v.raw_q1_taken)+' / '+n(v.raw_q1_mean)],['shaped Q1 执行动作 / 策略均值',n(v.shaped_q1_taken)+' / '+n(v.shaped_q1_mean)],['继承H8任务Reward（本tick）',number(Object.entries(modelDelta).filter(([k])=>!['control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward'].includes(k)).reduce((s,[,v])=>s+v,0))],['JE10总Reward（本tick）',number(sum(modelDelta))],['历史训练rollout Reward','未记录；本页为离线Web诊断'],['计算来源',m.origin],['Critic SHA-256',m.critic_sha256],['Reward ledger SHA-256',m.reward_ledger_sha256],['覆盖率',m.measured_count+'/'+doc.actions.length+' 决策tick；无缺测；全部合法联合动作']]);
    }
    if(doc.meta.approximate_a1){
      if(!approxPanel.isConnected)comparison.after(approxPanel);
      const m=doc.meta.approximate_a1,v=diagnostic?.approximate_a1,n=x=>x==null?'未记录':x.toFixed(8);
      $('je10-advantage-status').textContent=m.limitations+' 公式：'+m.definition+'；任务代理奖励：'+m.reward_definition;
      details($('je10-advantage-details'),[['raw A1（近似）',n(v?.raw_a1)],['shaped A1（近似）',n(v?.shaped_a1)],['ΔA = shaped A1 − raw A1（近似）',n(v?.delta_a)],['raw A1 相邻 tick 增量',n(v?.raw_a1_increment)],['shaped A1 相邻 tick 增量',n(v?.shaped_a1_increment)],['ΔA 相邻 tick 增量',n(v?.delta_a_increment)],['raw Q1 执行动作 / 策略均值',n(v?.raw_q1_taken)+' / '+n(v?.raw_q1_mean)],['shaped Q1 执行动作 / 策略均值',n(v?.shaped_q1_taken)+' / '+n(v?.shaped_q1_mean)],['严格训练 raw / shaped A1','未记录：没有历史训练 rollout 与 hidden aux；这里 V=0，不调用未经验证 Critic'],['覆盖率',m.measured_count+'/'+doc.actions.length+' tick；全部合法 move×ability 反事实，按同 checkpoint 策略加权']]);
    }
    if(diagnostic){
      const box=document.createElement('div'),decision=diagnostic.decision;
      const offline=diagnostic.offline_recomputed_current_reward;
      if(decision){
      details(box,[['本步状态 SHA-256（前 → 后）',`${diagnostic.state_before_sha256} → ${diagnostic.state_after_sha256}`],['合法联合动作数（含道具）',String(decision.legal_count)],['移动策略概率（上/下/左/右/停）',decision.move_probability.map(v=>(100*v).toFixed(3)+'%').join(' / ')],['技能策略概率（无/泡/道具）',decision.ability_probability.map(v=>(100*v).toFixed(3)+'%').join(' / ')],['当前逻辑近似总 Reward（模型 / 对手）',offline?offline.approx_total_reward.map(number).join(' / '):'未记录'],['训练时 Reward / 严格 A₁','未记录；当前近似重算不代表训练时账本']]);
      if(doc.meta.approximate_a1){const r=diagnostic.route_diagnostic;box.append(text('p',`可观测路线（近似）：${r.phase} / ${JSON.stringify(r.target)}；距离 ${diagnostic.distance??'未记录'} / oracle ${JSON.stringify(diagnostic.oracle)}；连续 Phi ${r.phi_before.toFixed(8)} → ${r.phi_after.toFixed(8)}；奖励分量 ${JSON.stringify(diagnostic.approximate_reward_components)}；原始事件 ${JSON.stringify(diagnostic.events)}。训练 Reward 完整分量未记录；仅展示声明的任务代理/控制/路线分量。`));}
      }else{
        const previous=doc.diagnostic_ticks?.[t-2],approx=diagnostic.approximate_reward_gae;
        details(box,[['本步状态 SHA-256（前 → 后）',`${diagnostic.state_before_sha256||'未记录'} → ${diagnostic.state_after_sha256||diagnostic.state_hash}`],['路线阶段 / 目标',`${diagnostic.phase} / ${JSON.stringify(diagnostic.target)}`],['合法 oracle 移动集合',JSON.stringify(diagnostic.oracle)],['连续距离 / 势能',`${diagnostic.distance??'未记录'} / ${number(diagnostic.phi)}`],['糖泡事件 ID / 原施害者',`${diagnostic.control_event_id??'未记录'} / ${diagnostic.control_owner??'未记录'}`],['可观测奖励分解（近似）',JSON.stringify(diagnostic.approximate_reward_components)],['零值函数 Reward-GAE（近似）',approx==null?'未记录':number(approx)],['近似 GAE 变化',approx==null||previous?.approximate_reward_gae==null?'未记录':number(approx-previous.approximate_reward_gae)],['口径',diagnostic.approximation||'未记录；近似 GAE 不是反事实 Q/A₁']]);
      }
      if(offline){
        const route=offline.route[0];
        box.append(text('p',`offline_recomputed_current_reward / 近似诊断 · 当前路线 phase=${route.route_phase}，target=${route.route_target_identity}，来源=${route.route_target_provenance}；连续 Phi ${number(route.route_phi_before)} → ${number(route.route_phi_after)}，ΔPhi ${number(route.route_potential_delta)}，PBRS ${number(route.route_shaped_reward)}。`));
        if(offline.approx_phi_jump!==null&&Math.abs(offline.approx_phi_jump)>1e-6)box.append(text('p',`近似诊断异常：本步 Phi_before 与上一步 Phi_after 相差 ${offline.approx_phi_jump.toFixed(8)}；保留当前账本原值，未平滑。`,'return-warning'));
        box.append(text('p','原始动作与事件保持真实；近似奖励分解见上方。Value、Logit 和 Reward 均未用作严格 A₁；本次未计算 approx_value / approx_A₁。'));
      }
      $('state-details').append(box);
    }
    $('current-events').replaceChildren();const events=doc.events.filter(e=>e.tick===t);
    if(events.length)$('current-events').append(text('p',rewardEvidence(t),'event-reward'));
    for(const e of events)$('current-events').append(text('div',`${labels[e.kind]||e.kind} · ${e.player===0?'模型 P0':'对手 P1'}${e.origin===undefined?'':` · 源队${e.origin}`}${e.to?` → ${e.to}`:''}`,'event-line'));if(!events.length)$('current-events').append(text('span','—','empty'));
    chart();
    daggerTick();
  }
  function returnTick(){
    const t=player.sim.t,p=returnAnalysis.points[t],row=returnAudit?.rows.find(r=>r.tick===t);
    returnPanel.hidden=!doc.route;
    returnWarning.hidden=!p?.kind;returnWarning.textContent=p?.kind?F.describe(p):'';
    $('return-status').textContent=F.describe(p);
    const move=a=>a==null?'未记录':F.MOVES[a]+' ['+a+']',prob=v=>(v*100).toFixed(2)+'%';
    const fields=[['目标识别 ≠ 有效靠近','返程标签只表示持敌包与目标；距离下降才是观测进展'],['当前网格最短路首步',p?.path_first_moves.length?p.path_first_moves.map(move).join(' / '):'未记录'],['本 tick 执行动作（s[t−1] → s[t]）',t?`${move(doc.actions[t-1][0])} / ability ${doc.actions[t-1][1]}`:'开局'],['坐标口径','[行 y, 列 x]；上/下改变 y，左/右改变 x']];
    if(row){fields.push(['位置（前 → 后）',`${row.position_before.join(', ')} → ${row.position_after.join(', ')}`],['合法 move / ability',`${row.move_mask.map((v,i)=>v?move(i):null).filter(Boolean).join('、')} / ${row.ability_mask.flatMap((v,i)=>v?[i]:[]).join(', ')}`],['策略概率口径','同一 v35 Actor 的合法 move×ability 两头概率乘积；greedy执行。概率不是优势或因果解释']);
      const next=returnAudit.rows.find(r=>r.tick===t+1);if(next)fields.push(['下一决策（从当前状态出发）',`${move(next.taken[0])} ${prob(next.move_probabilities[next.taken[0]])}；向下 ${prob(next.move_probabilities[1])}`]);
      fields.push(['合法联合动作数',String(row.candidates.length)],['原始 / 塑形 Q、A 与跨 Critic 差','未记录；以下候选只比较实际策略概率、一步位置和网格距离']);
      fields.push(['为什么距离不变','本区间按 floor(y)、floor(x) 取整格距离；y 在 1.80↔1.98 时仍属同一格，Phi 不变，路线项只剩折扣惩罚。这解释信号平台，不能证明策略选择的因果。'],['可行进展的生产模拟见证','从 tick176 状态连续向下两步：tick177 到 y=1.98、距离38；tick178 到 y=2.16、距离37。实际 tick178 却折返向上；不是碰撞截停，也不是目标坐标颠倒。']);
    }else fields.push(['Actor / 候选审计窗口',returnAudit?'仅 tick147–207 已采集；当前 tick 未记录':'本局未采集额外 Actor 审计']);
    details($('return-details'),fields);
    const table=$('return-candidates');table.replaceChildren();
    if(row){const head=document.createElement('tr');for(const label of ['合法联合动作','策略概率','一步位置 [y,x]','目标距离','原始 Q / 塑形 Q'])head.append(text('th',label));table.append(head);
      for(const c of row.candidates){const tr=document.createElement('tr');tr.classList.toggle('selected-action',c.action[0]===row.taken[0]&&c.action[1]===row.taken[1]);for(const v of [`${move(c.action[0])} / ${c.action[1]}${c.action[0]===row.taken[0]?' · 已执行':''}`,prob(c.probability),c.position.map(v=>v.toFixed(3)).join(', '),`${row.distance_before}→${c.distance}`,'未记录 / 未记录'])tr.append(text('td',v));table.append(tr);}
      table.append(text('caption','冻结同一状态、对手动作与随机流的一步生产模拟；非法动作排除。网格距离不是 Q。'));
    }
    const audit=returnAudit?.advantage_audit;$('return-missing').replaceChildren();
    if(audit){for(const role of ['original','shaped']){const c=audit[role];$('return-missing').append(text('p',`${role==='original'?'H8 v34 g32 原始任务':'H8 v35 g32 塑形'} Critic · checkpoint ${c.checkpoint} · Critic ${c.critic}`));}for(const link of audit.missing_links)$('return-missing').append(text('p',link));const a=text('a','查看 tick147–207 全量动作 / 概率 / 路径及来源哈希');a.href='review-evidence/route-g32-return.json';a.target='_blank';a.rel='noopener';$('return-missing').append(a);}
    else $('return-missing').textContent='本局未采集额外训练链路审计';
    const original=A.atTick(doc,t),r=R.atTick(doc,t),gap=F.comparison(original,{advantage:r.shaped_advantage_a1});
    crossGap.textContent=doc.metrics?`同 Critic、两种奖励口径 ΔA(塑形−原始)：${R.signed(M.atTick(doc,t).delta_a)}；相邻 tick 增量：${R.signed(M.atTick(doc,t).delta_a_increment)}。raw侧不是独立原始任务 Critic；路线奖励不等于优势。`:`跨 Critic 优势差 ΔA(塑形−原始)：${gap===null?'未记录':R.signed(gap)}。这与相邻决策优势增量 A₁(t)−A₁(t−1) 分开；缺任一真实 A₁ 时不计算。`;
  }
  function routeTick(){
    const t=player.sim.t,r=R.atTick(doc,t),m=doc.meta.route,v=R.value,advantageBlocked=doc.meta.advantage_audit?.status==='blocked',av=n=>n===null||n===undefined?(advantageBlocked?'无法严格复算':v(n)):v(n);
    $('route-status').textContent=(r.route_phase==='NONE'?'无激活路线目标；未满足合法库存窗口或归因散包条件，距离不展示。 ':'')+(m?.kind==='fixture'?'测试 fixture · 非真实训练录像。':'')+(r.route_phase===null?'本 tick 路线：未记录':R.label(R.PHASES,r.route_phase))+` · γ ${v(m?.gamma)} · λ ${v(m?.lambda)}`+(m?.limitations?' · '+m.limitations:'');
    const fields=[['目标阶段 · route_phase',R.label(R.PHASES,r.route_phase)],['当前目标 · target_type',r.route_phase==='NONE'?'无激活目标':R.label(R.TARGETS,r.target_type)],['目标来源 · target_provenance',R.label(R.PROVENANCE,r.target_provenance)],['脱敏目标 ID · target_id',v(r.target_id)],['网格加权最短路（前 → 后）',r.route_phase==='NONE'?'不适用（无激活目标）':`${v(r.distance_before)} → ${v(r.distance_after)}`],['路线势能（前 → 后） · Phi',`${v(r.phi_before)} → ${v(r.phi_after)}`],['路线势能变化 · route_potential_delta',R.signed(r.route_potential_delta)],['路线奖励 · route_shaped_reward',R.signed(r.route_shaped_reward)],['拓扑区域（前 → 后）',`${v(r.topology_region_before)} → ${v(r.topology_region_after)}`],['开墙 / 拓扑增益 · topology_gain',v(r.topology_gain)],['阶段转换 · phase_transition',v(r.phase_transition)],['目标失效 · invalidated',v(r.invalidated)],['终态边界修正 · terminal_correction',v(r.terminal_correction)],['安全拒绝 · safety_reject_reason',R.reason(r.safety_reject_reason)]];
    details($('route-details'),fields);
    if(doc.metrics)$('route-details').append(text('p','Reward 为同版本封存账本在本局Web真实状态/事件上的离线复算；历史训练rollout Reward 未记录。路线项已含在JE10总Reward中，不重复求和。NONE表示无激活目标，不使用己笼距离代替。'));
    else if(doc.meta.diagnostic)$('route-details').append(text('p',`${doc.meta.offline_recomputed_current_reward?'当前 Reward 是 offline_recomputed_current_reward / 近似诊断；':'原环境 Reward：未记录；'}训练时总 Reward：未记录；本 tick 路线辅助 Reward：${R.signed(r.route_shaped_reward)}。路线项已包含在近似总奖励中，不重复求和。`));
    $('shaped-status').textContent=doc.metrics?'同checkpoint训练总Reward Critic · 离线Web反事实一步近似；raw侧共用本Critic，非独立原始任务Critic':m?.shaped_critic?`塑形回报训练 Critic · ${m.shaped_critic.critic_sha256.slice(0,12)} · 原始任务优势保持独立`:advantageBlocked?'无法严格复算：同代 Critic 与反事实训练奖励链未能同时证明':doc.meta.diagnostic?'未记录：尚无严格反事实结果，不能计算 A₁':'未记录：尚无经过验证的塑形训练 Critic';
    details($('shaped-details'),[['执行动作 Q₁ · shaped_critic_q1_taken',av(r.shaped_critic_q1_taken)],['策略平均 Q₁ · shaped_critic_q1_policy_mean',av(r.shaped_critic_q1_policy_mean)],['塑形一步优势 · shaped_advantage_a1',av(r.shaped_advantage_a1)],['塑形优势增量 · shaped_advantage_delta',av(r.shaped_advantage_delta)],['合法联合动作数',av(r.shaped_legal_count)]]);
    const option=$('advantage-mode').querySelector('[value="shaped"]');option.disabled=!m?.shaped_critic;$('advantage-mode').querySelector('[value="task"]').textContent=doc.metrics?'原始任务Reward（同Critic近似）':'原始任务Critic';
    option.textContent=m?.shaped_critic?'塑形 Critic':advantageBlocked?'塑形 Critic（无法严格复算）':'塑形 Critic（未记录）';
    if(option.disabled)$('advantage-mode').value='task';
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
  function rewardEvidence(tick,pid=Number($('reward-player').value)){const r=currentReward(tick,pid),a=doc.metrics&&pid===0?M.atTick(doc,tick):null;return (a?`模型 P0 raw A₁ ${R.signed(a.raw_a1)} · shaped A₁ ${R.signed(a.shaped_a1)} · ΔA ${R.signed(a.delta_a)} · `:'')+`${pid===0?'模型':'对手'} ${doc.meta.offline_recomputed_current_reward?'近似 ':''}Reward ${doc.meta.reward_status==='unrecorded'?'未记录':number(sum(r))}（同 tick 事件共用一次） · ${doc.meta.reward_status==='unrecorded'?'原环境 / 训练总 Reward 未采集':Object.entries(r).map(([k,v])=>`${V.reward(k).name} ${number(v)}`).join('；')||'无非零分量'}`;}
  function marker(track,tick,color,title,width){const b=text('button','','mark'+(width?' phase-mark':''));b.dataset.tick=tick;b.style.left=`${tick/doc.actions.length*100}%`;b.style.background=color;b.title=`tick ${tick} · ${P.time(tick)} · ${title}`;b.setAttribute('aria-label',b.title);if(width)b.style.width=`${width/doc.actions.length*100}%`;b.onclick=e=>{e.stopPropagation();seek(tick);};$(track).append(b);return b;}
  function numericTrack(track,entries,format,dense=false){
    const lanes=[],width=$('timeline-scroll').clientWidth*Number($('zoom').value);let lastLabel=-1000,lastWidth=0;
    for(const entry of entries){const x=entry.tick/doc.actions.length*width;
      const b=marker(track,entry.tick,entry.value<0?'#603b43':'#254c41',entry.title);b.textContent=format(entry.value,entry);b.classList.add('numeric-mark');
      if(dense){const labeled=x-lastLabel>=Math.max(86,lastWidth+6);b.classList.add(labeled?'dense-label':'dense-point');if(labeled){lastLabel=x;lastWidth=b.offsetWidth;}else b.textContent='';b.style.top=labeled?'6px':'31px';b.dataset.value=entry.value;continue;}
      let lane=lanes.findIndex(end=>x>=end+4);if(lane<0)lane=lanes.length;lanes[lane]=x+b.offsetWidth;
      b.style.top=`${7+lane*22}px`;b.dataset.value=entry.value;
    }
    const height=dense?48:Math.max(36,lanes.length*22+14);$(track).style.height=height+'px';document.querySelector(`[data-track-label="${track}"]`).style.height=height+'px';
  }
  function timeline(){
    if(!doc)return;
    for(const id of ['ruler','phase-track','event-track','failure-track','reward-track','advantage-track','advantage-delta-track','route-potential-track','route-reward-track','shaped-advantage-track','shaped-delta-track','dagger-track','annotation-track'])$(id).replaceChildren();
    $('timeline').style.width=`${Number($('zoom').value)*100}%`;
    for(let t=0;t<=doc.actions.length;t+=200){const e=text('span',P.time(t).slice(0,5));e.style.left=`${t/doc.actions.length*100}%`;$('ruler').append(e);}
    let start=0,phase='战斗';for(const e of doc.events.filter(e=>e.kind==='phase'&&e.player===0)){marker('phase-track',start,phase==='持包返程'?'#77c6aa':phase==='运输窗口'?'#d4ab64':'#5ba6ba',phase,e.tick-start);start=e.tick;phase=e.to;}marker('phase-track',start,'#5ba6ba',phase,doc.actions.length-start);
    const eventGroups=new Map();for(const e of doc.events.filter(e=>e.kind!=='phase'&&e.kind!=='bomb')){if(!eventGroups.has(e.tick))eventGroups.set(e.tick,[]);eventGroups.get(e.tick).push(e);}
    numericTrack('event-track',[...eventGroups].sort(([a],[b])=>a-b).map(([tick,events])=>({tick,value:sum(currentReward(tick)),title:events.map(e=>`${e.player===0?'模型':'对手'} ${labels[e.kind]||e.kind}`).join('；')+' · '+rewardEvidence(tick)})),v=>doc.meta.reward_status==='unrecorded'?'R 未记录':(doc.meta.offline_recomputed_current_reward?'近似 R ':'R ')+(v>0?'+':'')+number(v));
    for(const interval of returnAnalysis.intervals){const name=interval.kind==='return_oscillation'?'返程振荡失败':'返程停滞失败';const b=marker('failure-track',interval.start,'#8f4652',`${name} · tick ${interval.start}–${interval.end}（原始重复位置从 ${interval.observed_start} 起） · 网格距离 ${interval.distance} · 点击区间位置跳转`,interval.end-interval.start+1);b.textContent=name;b.classList.add('failure-mark');b.dataset.end=interval.end;b.onclick=e=>{e.stopPropagation();const rect=b.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width));seek(Math.round(interval.start+fraction*(interval.end-interval.start)));};}
    if(!returnAnalysis.intervals.length)$('failure-track').append(text('span',doc.route?'未达到判定条件':'本局未采集路线，未判定','dagger-empty'));
    const rewardPid=Number($('reward-player').value);for(const r of doc.rewards){const components=r[rewardPid+1];if(Object.keys(components).length)marker('reward-track',r[0],sum(components)<0?'#e78f94':'#7bcaac',`${rewardPid===0?'模型':'对手'}奖励增量 ${number(sum(components))} · ${Object.entries(components).map(([k,v])=>`${V.reward(k).name} ${number(v)}`).join('；')}`);}
    for(const a of rows)marker('annotation-track',a.tick,'#e192a6',a.note||a.category);
    D.renderTrack($('dagger-track'),doc,seek,P.time);
    for(const [track,key]of [['route-potential-track','route_potential_delta'],['route-reward-track','route_shaped_reward'],['shaped-advantage-track','shaped_advantage_a1'],['shaped-delta-track','shaped_advantage_delta']]){
      const entries=(doc.route?.ticks||[]).map(tick=>({tick,value:R.atTick(doc,tick)[key]})).filter(e=>e.value!==null).map(e=>({...e,title:`${key} ${R.signed(e.value)} · ${R.summary(doc,e.tick)} · ${rewardEvidence(e.tick,0)}`}));
      numericTrack(track,entries,(v,e)=>R.signed(v)+' '+(R.PHASES[R.atTick(doc,e.tick).route_phase]||'阶段未记录'),entries.length>200);
      if(!entries.length)$(track).append(text('span','未记录','dagger-empty'));
    }
    for(const [track,key]of [['advantage-track','advantage'],['advantage-delta-track','delta']]){
      const entries=(doc.metrics?doc.actions.map((_,i)=>i+1):doc.advantages?.ticks||[]).map(tick=>({tick,value:A.atTick(doc,tick)[key]})).filter(e=>e.value!==null).map(e=>({...e,title:`${key==='delta'?'优势增量（相邻决策 A₁ 之差）':'策略加权一步优势（执行 Q₁ 减合法策略平均 Q₁）'} · ${key} ${number(e.value)} · move × ability 联合动作`}));
      numericTrack(track,entries,v=>(v>0?'+':'')+number(v),true);if(!entries.length)$(track).append(text('span','未记录','dagger-empty'));
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
  function draw(){if(renderer&&player){const nextAction=doc.actions[player.sim.t],intents=nextAction?[nextAction[0],nextAction[2]]:[4,4];const motion=V.motion(player.sim,preview.sim,playing?accumulator/100:0,playing&&$('smooth').checked,doc.events,intents);motion.intents=intents;motion.labelScale=Math.max(1,Math.min(1.5,500/$('game').getBoundingClientRect().width));if($('advantage-overlay').checked){const r=R.atTick(doc,player.sim.t);motion.playerAdvantages=[$('advantage-mode').value==='shaped'?{advantage:r.shaped_advantage_a1,delta:r.shaped_advantage_delta,kind:'shaped'}:A.atTick(doc,player.sim.t),null];}if($('route-overlay').checked)motion.playerRoutes=[R.overlay(doc,player.sim.t),null];renderer.render(player.sim,player.sim.t*100+motion.fraction*100,motion);$('refresh-status').textContent=`实际 ${actualFPS} FPS · 模拟 10 tick/s · α ${motion.fraction.toFixed(3)} · 目标 ${$('frame-rate').value} FPS`;}}
  async function load(id,tick){
    const token=++loadToken;playing=false;$('canvas-loading').hidden=false;$('canvas-loading').textContent='正在验证录像与状态 hash';$('integrity').textContent='验证中';
    const row=[...catalog.replays,...(catalog.history_replays||[])].find(r=>r.id===id);if(!row)throw Error('本录像不在合格诊断清单：缺少完整 Reward / Critic 优势证据；原资产保留供审计，未跳转其他录像。');
    let bytes=row.localBytes;
    if(!bytes){const response=await fetch(`review-replays/${row.file}`);if(!response.ok)throw Error(`录像 HTTP ${response.status}`);bytes=new Uint8Array(await response.arrayBuffer());}
    if(bytes.length!==row.bytes||await P.sha256(bytes)!==row.sha256)throw Error('录像文件 hash / 大小不匹配');
    const next=P.validate(JSON.parse(new TextDecoder().decode(bytes)));const mapResponse=await fetch('assets/maps/levels.json');if(!mapResponse.ok)throw Error('地图 HTTP');const level=(await mapResponse.json()).find(x=>x.qqt_id===806);if(await P.hash(level)!==next.meta.map_sha256)throw Error('地图 hash');
    const versioned=next.meta.simulator_sha256===P.ROUTE_SIM_HASH,simPath=versioned?'review-sim/route-v35.js':'sim.js';
    const sr=await fetch(simPath);if(!sr.ok)throw Error('模拟器 HTTP');const simBytes=await sr.arrayBuffer();if(await P.sha256(simBytes)!==next.meta.simulator_sha256)throw Error('模拟器版本不匹配');
    if(versioned){const current=window.QQT;try{await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=simPath;script.onload=resolve;script.onerror=()=>reject(Error('模拟器加载失败'));document.head.append(script);});P.registerSimulator(P.ROUTE_SIM_HASH,window.QQT);}finally{window.QQT=current;}}
    const nextPlayer=new P.Player(next,level);await nextPlayer.verify();const assets=await QQTVisual.loadAssets(level);if(token!==loadToken)return;
    let nextReturnAudit=null;const ar=row.diagnostic_audit;
    if(ar){if(ar.file!=='review-evidence/route-g32-return.json')throw Error('未知返程审计文件');const response=await fetch(ar.file);if(!response.ok)throw Error('返程审计 HTTP');const raw=new Uint8Array(await response.arrayBuffer());if(raw.length!==ar.bytes||await P.sha256(raw)!==ar.sha256)throw Error('返程审计 hash');nextReturnAudit=F.validateAudit(JSON.parse(new TextDecoder().decode(raw)),next,row.sha256);}
    const nextAnalysis=F.analyze(next,level,P);if(nextReturnAudit&&JSON.stringify(nextReturnAudit.intervals)!==JSON.stringify(nextAnalysis.intervals))throw Error('返程失败区间重算不一致');if(token!==loadToken)return;
    Object.defineProperty(next,'asset_sha256',{value:row.sha256});
    doc=next;returnAnalysis=nextAnalysis;returnAudit=nextReturnAudit;player=nextPlayer;preview=new P.Player(doc,level);preview.cache=player.cache;rewardByTick=new Map(doc.rewards.map(r=>[r[0],r]));curveCache=null;renderer=QQTVisual.createRenderer($('game'),level,assets);rows=[];editing=null;$('annotation-form').hidden=true;
    try{const saved=localStorage.getItem(storageKey());if(saved)rows=P.validateAnnotations(JSON.parse(saved),doc);}catch(_){say('已有标注无效或存储不可用');}
    $('episode-title').textContent=(doc.meta.route?.kind==='fixture'?'合成测试 fixture · ':'')+`模型 P0 · ${doc.meta.model||doc.meta.id} 对阵 对手 P1`;$('episode-meta').textContent=`${doc.meta.scene_name||'历史抢包对战'} · qqt_id ${doc.meta.map} · ${doc.meta.scene_rules||doc.meta.variant} · 对手 ${doc.meta.opponent} · seed ${doc.meta.seed} · checkpoint ${doc.meta.checkpoint_sha256.slice(0,12)} · map ${doc.meta.map_sha256.slice(0,12)}`;$('terminal').textContent=doc.summary.done===false?'片段结束：对局仍在进行，未记录胜负':`终局：${doc.summary.winner===null?'平局':doc.summary.winner===0?'模型胜':'对手胜'}`;
    const metrics=Boolean(doc.metrics);
    document.querySelector('.route-explanation').textContent=metrics?'本批raw侧是继承H8任务Reward、shaped侧是JE10总Reward；两侧使用同一个经过总Reward训练的Critic做离线一步反事实近似。路线奖励是人工辅助量，不能当作优势。非历史训练rollout，非独立原始任务Critic。':'原始任务Critic估计任务回报；路线奖励是人工辅助量。塑形Critic只有经过塑形回报训练并验证才展示；路线奖励不能代替优势。';
    const fixedEight=doc.meta.batch_id==='v36-fixed-eight-20261017',je10=doc.meta.batch_id==='je10-best-completed-20261018';
    $('route-experiment').hidden=!doc.meta.diagnostic||fixedEight||je10||metrics;
    const reference=catalog.model_comparison?.models.find(m=>m.checkpoint_sha256===doc.meta.checkpoint_sha256);modelSummary.hidden=!reference;if(reference){modelSummary.replaceChildren(text('h3',`${reference.id} · ${reference.label}`),text('p',`source run: ${reference.source_run} · g${reference.generation} · Actor ${reference.checkpoint_sha256.slice(0,12)} · Critic ${reference.critic.critic_sha256.slice(0,12)}`),text('p','固定同协议评估参考，非本页小样本胜率；默认综合发布基线仍为r01 g8。Python Tactical与JS Hunter分别报告。'));details(modelSummary.appendChild(document.createElement('div')),[['Python Tactical 安全击杀 / 自伤',`${reference.tactical.safe_win}/64 · 自伤 ${reference.tactical.self_episodes}/64`],['JS Hunter 难 / 中 / 易',Object.values(reference.hunter).map(v=>`${v.safe_win}/64（自伤${v.self_episodes}）`).join(' / ')],...Object.entries(reference.route).map(([k,v])=>[({native_death_stock:'死亡库存窗口',native_sugar_stock:'糖泡库存窗口',real_drop:'真实掉包',carry_near:'预置近程返程',carry_mid:'预置中程返程',carry_far:'预置远程返程',recovery_threat:'受威胁恢复',natural_full2400:'自然完整局'})[k],`拾包 ${v.pickup_episodes}/16 · 交付 ${v.delivery_episodes}/16 · 死亡 ${v.death} · 自伤 ${v.self_death}`])]);}
    if(reference){const evidence=text('p','');for(const [label,file]of [['固定评估 / Actor / Critic / Reward 配套凭证',catalog.model_comparison.selection_file],['全部预声明录像、复算与尺寸','review-evidence/good-models-20261019/verification.json']]){const link=text('a',label);link.href=file;link.target='_blank';link.rel='noopener';evidence.append(link,document.createElement('br'));}modelSummary.append(evidence);}
    let warning=$('episode-warning');if(!warning){warning=text('p','','diagnostic-warning');warning.id='episode-warning';document.querySelector('.episode-head').after(warning);}warning.hidden=!doc.meta.diagnostic;warning.textContent=metrics?`训练指标诊断载体 · ${({'original_trajectory_recomputed':'原录像同轨迹离线复算','same_checkpoint_new_seed':'同checkpoint新seed重跑','best_model_replacement':'当前综合发布最佳模型替代重跑'})[doc.meta.metrics.origin]} · ${doc.meta.scene_rules} · 全部预声明seed保留；非历史训练rollout。完整同版本Reward账本在Web真实事件上复算；同checkpoint真实Critic离线一步近似（不是V=0）。raw侧与shaped侧共享total-reward Critic，非两个独立训练Critic。终局若仍有危险，未结算奖励保留为账本事实。`:je10?`固定四场全部保留、不择优 · ${doc.meta.scene_rules} · 真实对手 ${doc.meta.opponent} · 已完成训练 checkpoint 的异步评估诊断，非历史训练 rollout。raw/shaped A1 为零后续值、策略加权反事实一步近似；严格训练 Reward / A1 未记录。`:fixedEight?`固定八局全部保留 · ${doc.meta.scene_rules} · 真实对手 ${doc.meta.opponent} · offline_recomputed_current_reward / 近似诊断，非 training-time reward；严格 A₁ 未计算`:doc.meta.diagnostic?'诊断场景，不代表自然完整对局成功 · 预置掉包 / assist-off / 固定向上对手':'';
    document.querySelector('.reward-total > span').textContent=fixedEight?'近似奖励增量 / 累计':'奖励增量 / 累计';
    document.querySelector('.component-head > span').textContent=fixedEight?'当前逻辑近似奖励分量 · 本 tick / 累计':'奖励分量 · 本 tick / 累计';
    const rewardTrackLabel=[...document.querySelectorAll('.track-label strong')].find(e=>e.textContent==='Reward 轨'||e.textContent==='近似 Reward 轨');
    if(rewardTrackLabel)rewardTrackLabel.textContent=fixedEight?'近似 Reward 轨':'Reward 轨';
    $('seek').max=doc.actions.length;$('duration').textContent=P.time(doc.actions.length);$('integrity').style.color='#82d2b6';
    curveOptions();daggerEpisode();
    if(daggerCatalog){const related=daggerCatalog.runs.find(r=>r.checkpoint_sha256===doc.meta.checkpoint_sha256);if(related)$('dagger-run').value=related.id;await daggerSummary($('dagger-run').value);}
    if(token!==loadToken)return;
    details($('teacher'),['eligible','query','accept','reject'].map(k=>[k,doc.meta.teacher?.[k]??'未测量']));
    $('provenance').replaceChildren();for(const [k,v] of [['Episode',doc.meta.episode_id],['Checkpoint SHA',doc.meta.checkpoint_sha256],['Model SHA',doc.meta.model_sha256],['Simulator SHA',doc.meta.simulator_sha256],['Opponent SHA',doc.meta.opponent_sha256],['Replay SHA',row.sha256],['Reward',doc.meta.reward_source||'未测量'],['原始 / gzip',`${row.bytes} / ${row.gzip_bytes} bytes`]])$('provenance').append(text('div',`${k}: ${v}`));
    if(doc.meta.route)for(const [k,v]of Object.entries(doc.meta.route).filter(([k])=>k.endsWith('_sha256'))) $('provenance').append(text('div',`路线 ${k}: ${v}`));
    if(doc.meta.metrics)for(const [k,v]of Object.entries(doc.meta.metrics).filter(([k])=>k.endsWith('_sha256'))) $('provenance').append(text('div',`${k}: ${v}`));
    if(doc.meta.critic_sha256)$('provenance').append(text('div',`同代塑形 Critic NPZ SHA: ${doc.meta.critic_sha256}`));
    for(const id of ['play','step-back','step-next','seek','replay-start','copy-position','copy-link','add-annotation','export-annotations','import-annotations'])$(id).disabled=false;
    document.querySelector('.timeline-layout').hidden=false;
    $('canvas-loading').hidden=true;list();timeline();annotations();seek(Math.min(doc.actions.length,Math.max(0,Number.isInteger(tick)?tick:0)));$('integrity').textContent='hash 已验证';say('录像与间隔状态 hash 校验通过');
  }
  function fail(e){playing=false;document.querySelector('.timeline-layout').hidden=true;player=null;preview=null;renderer=null;doc=null;$('locator').value='';$('exact-time').textContent='无法定位：录像不可用';for(const id of ['state-details','advantage-details','shaped-details','je10-advantage-details','route-details','policy-decision-details','model-summary','components','provenance','current-events'])$(id).replaceChildren();$('reward-delta').textContent='未记录';$('reward-total').textContent='未记录';$('episode-title').textContent='录像不可用';$('game').getContext('2d').clearRect(0,0,$('game').width,$('game').height);for(const id of ['play','step-back','step-next','seek','replay-start','copy-position','copy-link','add-annotation','export-annotations','import-annotations'])$(id).disabled=true;say(e.message);$('canvas-loading').hidden=false;$('canvas-loading').textContent=e.message;$('integrity').textContent='验证失败';}
  const keyboardStep=ReviewKeyboard.bindReviewKeyboard({onStep:(direction,amount)=>{
    if(player)seek(ReviewKeyboard.steppedTick(player.sim.t,direction,amount,doc.actions.length));
  }});
  document.addEventListener('keydown',keyboardStep);
  $('play').onclick=()=>{if(!player)return;if(playing)pause();else start();hashPosition();};
  $('replay-start').onclick=()=>{seek(0);start();};$('step-back').onclick=()=>seek(Math.max(0,player.sim.t-1));$('step-next').onclick=()=>seek(Math.min(doc.actions.length,player.sim.t+1));$('seek').oninput=()=>seek(Number($('seek').value));$('filter').oninput=()=>list();$('zoom').oninput=()=>timeline();
  $('timeline').onclick=e=>{const rect=$('timeline').getBoundingClientRect();seek(Math.round(Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*doc.actions.length));};
  $('reward-chart').onclick=e=>{const rect=e.target.getBoundingClientRect();seek(Math.round((e.clientX-rect.left)/rect.width*doc.actions.length));};
  $('curve-component').onchange=chart;$('curve-mode').onchange=chart;$('reward-player').onchange=()=>{update();timeline();};$('english-rewards').onchange=()=>{curveOptions();update();};$('smooth').onchange=draw;$('advantage-overlay').onchange=draw;$('route-overlay').onchange=draw;$('advantage-mode').onchange=draw;
  $('dagger-run').onchange=()=>daggerSummary($('dagger-run').value).catch(e=>say(e.message));
  $('frame-rate').onchange=()=>{lastDraw=0;frameStart=performance.now();frameCount=0;$('refresh-status').textContent=`目标 ${$('frame-rate').value} fps · 逻辑 10 tick/s`;};
  $('advanced-close').addEventListener('click',pause);$('advanced-forget').addEventListener('click',pause);
  async function copy(value){try{await navigator.clipboard.writeText(value);say('已复制');}catch(_){$('locator').focus();$('locator').select();say('剪贴板不可用；定位已选中');}}
  $('copy-position').onclick=()=>copy(P.locator(doc,player.sim.t));$('copy-link').onclick=()=>{hashPosition();copy(location.href);};
  $('add-annotation').onclick=()=>form();$('cancel-annotation').onclick=()=>{$('annotation-form').hidden=true;};
  $('annotation-form').onsubmit=e=>{e.preventDefault();const a={id:editing||crypto.randomUUID(),tick:Number($('annotation-tick').value),category:$('annotation-category').value,severity:$('annotation-severity').value,note:$('annotation-note').value};const next=rows.filter(x=>x.id!==a.id).concat(a);try{P.validateAnnotations(P.annotationDocument(doc,next),doc);rows=next;persist();$('annotation-form').hidden=true;}catch(e){say(e.message);}};
  $('export-annotations').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(P.annotationDocument(doc,rows),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('import-replay').onclick=()=>$('replay-file').click();
  $('replay-file').onchange=async()=>{try{
    const file=$('replay-file').files[0];if(!file)return;if(!catalog)throw Error('目录尚未就绪');if(file.size>64*1024*1024)throw Error('录像文件过大');
    const bytes=new Uint8Array(await file.arrayBuffer()),next=P.validate(JSON.parse(new TextDecoder().decode(bytes))),m=next.meta;
    const row={id:m.id,seed:m.seed,scene:m.scene||'native',variant:m.variant||'native',scene_name:m.native_fixture?.task||m.scene_name||'原生完整对局',scene_rules:m.native_fixture?'声明的原生 reset fixture':'原生规则',model:m.model||m.id,opponent:m.opponent,checkpoint_sha256:m.checkpoint_sha256,map:m.map||806,map_sha256:m.map_sha256,diagnostic:Boolean(m.diagnostic),summary:next.summary,bytes:bytes.length,sha256:await P.sha256(bytes),localBytes:bytes};
    catalog.replays=catalog.replays.filter(r=>r.id!==m.id);catalog.replays.unshift(row);sceneFilter.value='';modelFilter.value='';$('filter').value='';list();await load(m.id,0);say('诊断录像已验证并导入本次会话');
  }catch(e){fail(e);}finally{$('replay-file').value='';}};
  $('import-annotations').onclick=()=>$('annotation-file').click();$('annotation-file').onchange=async()=>{try{const file=$('annotation-file').files[0];if(!file)return;if(file.size>1024*1024)throw Error('标注文件过大');const next=P.validateAnnotations(JSON.parse(await file.text()),doc);rows=next;persist();say('标注导入成功');}catch(e){say(`导入失败：${e.message}`);}finally{$('annotation-file').value='';}};
  window.addEventListener('hashchange',()=>{const p=new URLSearchParams(location.hash.slice(1)),id=p.get('replay'),tick=Number(p.get('tick')||0);if(!Number.isInteger(tick)||tick<0){say('非法深链 tick');return;}if(doc?.meta.id===id)seek(Math.min(tick,doc.actions.length));else load(id,tick).catch(fail);});
  window.addEventListener('resize',()=>{if(doc){timeline();draw();}});
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
  try{const r=await fetch('review-catalog.json',{cache:'no-store'});if(!r.ok)throw Error(`目录 HTTP ${r.status}`);catalog=await r.json();if(catalog.schema!=='qqt.diagnostic-catalog/v1')throw Error('目录 schema');
    const dr=await fetch('review-dagger/catalog.json');if(!dr.ok)throw Error('DAgger 目录 HTTP');daggerCatalog=await dr.json();if(daggerCatalog.schema!=='qqt.dagger-catalog/v1')throw Error('DAgger 目录 schema');
    for(const row of daggerCatalog.runs)$('dagger-run').add(new Option(row.title+' · '+row.scope,row.id));
    for(const s of catalog.scenes||[])sceneFilter.add(new Option(s.name,s.id));for(const m of catalog.model_comparison?.models||[])modelFilter.add(new Option(`${m.id} · ${m.label}`,m.id));
    $('count').textContent=catalog.replays.length;$('catalog-meta').textContent=`${catalog.replays.length} 局 · ${catalog.model}\n${catalog.scenes?.length||1} 种场景 · ${catalog.screened_episodes} 个固定 seed\n${catalog.raw_bytes} bytes / gzip ${catalog.gzip_bytes} bytes`;if(catalog.metrics_admission){const gate=text('p',`准入：${catalog.replays.length}局完整Reward/真实Critic指标；${catalog.metrics_admission.excluded_count}条旧缺测入口已由真实对局替代。`);const link=text('a','两个原始tick根因及不可变证据');link.href=catalog.metrics_admission.audit_file;link.target='_blank';link.rel='noopener';gate.append(document.createElement('br'),link);$('catalog-meta').after(gate);const audit=text('details','');audit.id='route-target-audit-summary';audit.append(text('summary','两个指定旧tick为何异常'),text('p','carry_home tick107：把己半场误标为入笼，且旧phase4误用掉包距离通道，Phi/路线奖励恒0。原生交付时仍有fuse23活泡，账本的24分需危险结束后结算，Web立即终局没有该结算tick。'),text('p','open tick436：模型空手、敌未受控，无激活路线目标；NONE分支却导出默认己笼距离4→4。新版NONE距离为空，不再误显示返程。原文件SHA及前后30tick数据保留，未手工改历史奖励。'));gate.after(audit);}list();const p=new URLSearchParams(location.hash.slice(1));await load(p.get('replay')||catalog.replays[0].id,Number(p.get('tick')||0));}catch(e){fail(e);}
})();
