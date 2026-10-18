'use strict';
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;else root.QQTReviewRoute=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const SCHEMA='qqt.route-potential/v1';
  const PHASES={NONE:'无目标',B_STOCK:'B · 窗口取库存',A_ENEMY_DROP:'A · 真实敌源掉包',OWN_RECOVERY:'己包回收（非敌源）',C_RETURN:'C · 持包返程',D_DELIVERY:'D · 入笼交付'};
  const TARGETS={enemy_cage:'敌笼库存',loose_bun:'散落包',home_cage:'己方交付笼'};
  const PROVENANCE={enemy_stock:'敌笼库存',enemy_carried_drop:'敌方携带死亡掉包',own_recovery:'己包回收',held_enemy_bun:'已持敌源包',own_cage:'己方笼'};
  const VALUES=['phi_before','phi_after','route_potential_delta','route_shaped_reward','shaped_critic_q1_taken','shaped_critic_q1_policy_mean','shaped_advantage_a1','shaped_advantage_delta'];
  const FLAGS=['phase_transition','invalidated','terminal_correction','terminal'];
  const COLUMNS=['ticks','route_phase','target','distance_before','distance_after','topology_region_before','topology_region_after','topology_gain','topology_evidence','safety_reject_reason','shaped_legal_count',...VALUES,...FLAGS];
  function signed(v){
    if(v===null||v===undefined)return '未记录';
    const [whole,fraction]=Math.abs(v).toFixed(6).split('.');
    return (v>=0?'+':'-')+whole+'.'+fraction.replace(/0+$/,'').padEnd(3,'0');
  }
  const value=v=>v===null||v===undefined?'未记录':v==='INF'?'不可达（INF）':typeof v==='boolean'?(v?'是':'否'):String(v);
  const label=(table,key)=>key==null?'未记录':Object.hasOwn(table,key)?`${table[key]} · ${key}`:`未翻译（${key}）`;
  const machine=v=>typeof v==='string'&&/^[a-z][a-z0-9_]{0,63}$/.test(v);
  const hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
  const REASONS={unsafe:'安全核验未通过',self_death:'自身死亡',self_trap:'自身受困',teammate_harm:'队友受伤',rescue_blocked:'救援路线被堵',cage_blocked:'笼路径被堵',lost_carried_bun:'携带包丢失',unreachable:'目标不可达',target_invalidated:'目标失效',window_closed:'合法窗口关闭'};
  const reason=key=>key==null?'未记录 / 无拒绝记录':Object.hasOwn(REASONS,key)?`${REASONS[key]} · ${key}`:`未翻译（${key}）`;
  function validate(doc){
    const m=doc.meta.route,d=doc.route,fail=()=>{throw Error('route schema / evidence / recompute');};
    if(m==null&&d==null)return;
    if(!m||!d||m.schema!==SCHEMA||d.schema!==SCHEMA||!['fixture','production'].includes(m.kind)||m.status!=='recorded'||
      !Number.isFinite(m.gamma)||m.gamma<0||m.gamma>1||!Number.isFinite(m.lambda)||m.lambda<0||
      m.distance_metric!=='production_weighted_grid_v1'||!Number.isFinite(d.scale)||d.scale<=0||d.scale>1e-6||
      !/^[A-Za-z0-9_-]{1,120}$/.test(m.source_run)||!/^\d{4}-\d\d-\d\dT/.test(m.generated_at)||!Number.isFinite(Date.parse(m.generated_at)))fail();
    for(const [key,parent]of [['checkpoint_sha256','checkpoint_sha256'],['policy_sha256','model_sha256'],['simulator_sha256','simulator_sha256']])if(m[key]!==doc.meta[parent])fail();
    for(const key of ['reward_ledger_sha256','route_implementation_sha256','source_sha256','recomputed_sha256'])if(!hash(m[key]))fail();
    if(Object.keys(d).some(k=>!['schema','scale','targets',...COLUMNS].includes(k))||COLUMNS.some(k=>!Array.isArray(d[k])||d[k].length!==d.ticks.length)||d.ticks.length>doc.actions.length||!Array.isArray(d.targets)||d.targets.length>d.ticks.length)fail();
    for(const t of d.targets)if(!t||Object.keys(t).sort().join(',')!=='id,provenance,type'||!Object.hasOwn(TARGETS,t.type)||!Object.hasOwn(PROVENANCE,t.provenance)||!/^target-[a-f0-9]{64}$/.test(t.id))fail();
    const metaKeys=['schema','kind','status','gamma','lambda','checkpoint_sha256','policy_sha256','simulator_sha256','reward_ledger_sha256','route_implementation_sha256','source_sha256','recomputed_sha256','source_run','generated_at','distance_metric','shaped_critic','measured_count','missing_count','shaped_measured_count','shaped_missing_count','semantics','limitations'];
    if(Object.keys(m).some(k=>!metaKeys.includes(k)))fail();
    const diagnostic=m.semantics==='diagnostic-post-target-uncorrected/v1';
    if(m.semantics!==undefined&&(!diagnostic||doc.meta.diagnostic!==true||m.kind!=='production'||typeof m.limitations!=='string'||!m.limitations))fail();
    const c=m.shaped_critic;
    if(c!=null&&Object.keys(c).some(k=>!['schema','trained_on_route_reward','checkpoint_sha256','critic_sha256','reward_ledger_sha256','counterfactual_sha256','action_scope','legal_actions'].includes(k)))fail();
    if(c!=null&&(c.schema!=='qqt.route-shaped-a1/v1'||c.trained_on_route_reward!==true||c.checkpoint_sha256!==m.checkpoint_sha256||c.reward_ledger_sha256!==m.reward_ledger_sha256||!hash(c.critic_sha256)||!hash(c.counterfactual_sha256)||c.action_scope!=='joint_move_ability'||typeof c.legal_actions!=='string'||!c.legal_actions))fail();
    let count=0,measured=0,prior=null;
    for(let i=0;i<d.ticks.length;i++){
      const t=d.ticks[i];if(!Number.isInteger(t)||t<1||t>doc.actions.length||(i&&t<=d.ticks[i-1]))fail();
      if(d.route_phase[i]!==null&&!Object.hasOwn(PHASES,d.route_phase[i]))fail();
      if(d.target[i]!==null&&(!Number.isInteger(d.target[i])||!d.targets[d.target[i]]))fail();
      for(const k of VALUES)if(d[k][i]!==null&&!Number.isSafeInteger(d[k][i]))fail();
      for(const k of FLAGS)if(d[k][i]!==null&&typeof d[k][i]!=='boolean')fail();
      for(const k of ['distance_before','distance_after'])if(d[k][i]!==null&&d[k][i]!=='INF'&&(!Number.isFinite(d[k][i])||d[k][i]<0))fail();
      for(const k of ['topology_region_before','topology_region_after','safety_reject_reason'])if(d[k][i]!==null&&!machine(d[k][i]))fail();
      const evidence=d.topology_evidence[i];
      if(evidence!==null&&(!Array.isArray(evidence)||evidence.length!==3||evidence.some(v=>!hash(v))))fail();
      if(d.topology_gain[i]>0&&evidence===null)fail();
      if(d.route_phase[i]==='A_ENEMY_DROP'&&d.target[i]!==null&&d.targets[d.target[i]].provenance!=='enemy_carried_drop')fail();
      if(d.route_phase[i]==='B_STOCK'&&d.target[i]!==null&&d.targets[d.target[i]].provenance!=='enemy_stock')fail();
      if(d.topology_gain[i]!==null&&(!Number.isFinite(d.topology_gain[i])||d.topology_gain[i]<0))fail();
      const pb=d.phi_before[i],pa=d.phi_after[i],delta=d.route_potential_delta[i],reward=d.route_shaped_reward[i];
      // Quantization error from both Phi inputs and the result; no independent bonus.
      if(pb===null||pa===null){if(delta!==null||reward!==null)fail();}
      else if(delta===null||reward===null||Math.abs(delta-(m.gamma*pa-pb))>2||Math.abs(reward-m.lambda*delta)>1+m.lambda)fail();
      if(delta!==null)count++;
      if(i&&t===d.ticks[i-1]+1&&pb!==null&&d.phi_after[i-1]!==null&&Math.abs(pb-d.phi_after[i-1])>1){
        const current=doc.meta.offline_recomputed_current_reward&&doc.diagnostic_ticks?.[t-1]?.offline_recomputed_current_reward;
        const jump=current?.approx_phi_jump??doc.meta.metrics?.potential_discontinuities?.find(r=>r[0]===t)?.[1];
        if(!Number.isFinite(jump)||Math.abs(pb-d.phi_after[i-1]-jump/d.scale)>2)fail();
      }
      if(!diagnostic&&t===doc.actions.length&&pa!==null&&(pa!==0||d.terminal[i]!==true))fail();
      if(d.terminal[i]===true&&(t!==doc.actions.length||(!diagnostic&&pa!==0)))fail();
      if(d.terminal_correction[i]===true&&d.terminal[i]!==true)fail();
      if(d.topology_gain[i]>0&&(d.safety_reject_reason[i]!==null||d.invalidated[i]!==false||d.target[i]===null||!(
        (d.distance_before[i]==='INF'&&Number.isFinite(d.distance_after[i]))||
        (Number.isFinite(d.distance_before[i])&&Number.isFinite(d.distance_after[i])&&d.distance_after[i]<d.distance_before[i])||
        (d.topology_region_before[i]!==null&&d.topology_region_after[i]!==null&&d.topology_region_before[i]!==d.topology_region_after[i]))))fail();
      const q=d.shaped_critic_q1_taken[i],mean=d.shaped_critic_q1_policy_mean[i],a=d.shaped_advantage_a1[i],da=d.shaped_advantage_delta[i],n=d.shaped_legal_count[i];
      if(a===null){if(q!==null||mean!==null||da!==null||n!==null)fail();prior=null;continue;}
      measured++;
      if(!c||q===null||mean===null||Math.abs(a-(q-mean))>2||!Number.isInteger(n)||n<1||n>15)fail();
      if(prior===null||t!==prior.tick+1){if(da!==null)fail();}
      else if(da===null||Math.abs(da-(a-prior.a))>2)fail();
      prior={tick:t,a};
    }
    if(m.measured_count!==count||m.missing_count!==doc.actions.length-count||m.shaped_measured_count!==measured||m.shaped_missing_count!==doc.actions.length-measured||(!c&&measured))fail();
  }
  function atTick(doc,tick){
    const d=doc.route,i=d?.ticks.indexOf(tick)??-1,r={};
    for(const k of COLUMNS.filter(k=>!['ticks','target'].includes(k)))r[k]=i<0?null:d[k][i]===null?null:VALUES.includes(k)?d[k][i]*d.scale:d[k][i];
    const target=i<0||d.target[i]===null?null:d.targets[d.target[i]];
    if(r.route_phase==='NONE'){r.distance_before=null;r.distance_after=null;}
    if(doc.metrics){const i=tick-1,d=doc.metrics;for(const [k,c]of [['shaped_critic_q1_taken','shaped_q1_taken'],['shaped_critic_q1_policy_mean','shaped_q1_mean'],['shaped_advantage_a1','shaped_a1'],['shaped_advantage_delta','shaped_a1_increment']])r[k]=i<0||i>=doc.actions.length||d[c][i]===null?null:d[c][i]*d.scale;r.shaped_legal_count=d.legal_count[i]??null;}
    return {...r,target_type:target?.type??null,target_provenance:target?.provenance??null,target_id:target?.id??null};
  }
  function summary(doc,tick){
    const r=atTick(doc,tick),m=doc.meta.route;
    return `${r.route_phase==='NONE'?'无激活路线目标（未满足合法库存窗口或归因散包条件），无目标距离 · ':''}${label(PHASES,r.route_phase)} · 路线ΔΦ ${signed(r.route_potential_delta)} · 路线奖励 ${signed(r.route_shaped_reward)} · λ ${value(m?.lambda)} · 路程 ${value(r.distance_before)}→${value(r.distance_after)} · Phi ${value(r.phi_before)}→${value(r.phi_after)} · 目标 ${label(TARGETS,r.target_type)} / ${label(PROVENANCE,r.target_provenance)} / ${value(r.target_id)} · 区域 ${value(r.topology_region_before)}→${value(r.topology_region_after)} · 开墙增益 ${value(r.topology_gain)} · 拓扑证据 ${r.topology_evidence?.join(' / ')||'未记录'} · 阶段切换 ${value(r.phase_transition)} · 失效 ${value(r.invalidated)} · 终态修正 ${value(r.terminal_correction)} · 安全拒绝 ${reason(r.safety_reject_reason)}`;
  }
  function overlay(doc,tick){
    const r=atTick(doc,tick),distance=v=>v==='INF'?'∞':v===null?'?':String(v);
    return {text:`路线ΔΦ ${signed(r.route_potential_delta)}｜路程${distance(r.distance_before)}→${distance(r.distance_after)}`,value:r.route_potential_delta};
  }
  return {SCHEMA,PHASES,TARGETS,PROVENANCE,COLUMNS,VALUES,FLAGS,validate,atTick,signed,value,label,reason,summary,overlay};
});
