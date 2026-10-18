'use strict';
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.QQTReviewMetrics=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 const SCHEMA='qqt.sealed-ledger-a1/v1';
 const FIELDS=['raw_q1_taken','raw_q1_mean','raw_a1','shaped_q1_taken','shaped_q1_mean','shaped_a1','delta_a','raw_a1_increment','shaped_a1_increment','delta_a_increment'];
 const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
 function atTick(doc,tick){const d=doc.metrics,i=tick-1;return Object.fromEntries(FIELDS.map(k=>[k,!d||i<0||i>=doc.actions.length||d[k][i]===null?null:d[k][i]*d.scale]));}
 function validate(doc){
  if(!doc.meta.metrics&&!doc.metrics)return;
  const m=doc.meta.metrics,d=doc.metrics,n=doc.actions.length,fail=()=>{throw Error('metrics admission / sealed Critic ledger evidence');};
  if(!m||!d||m.schema!==SCHEMA||d.schema!==SCHEMA||m.status!=='offline_approximate_one_step'||d.scale!==1e-6||m.measured_count!==n||m.missing_count!==0||m.future_value!=='same_checkpoint_critic'||!Number.isFinite(m.gamma)||m.gamma<0||m.gamma>1||typeof m.limitations!=='string'||!m.limitations||m.action_scope!=='joint_move_ability')fail();
  for(const [k,v]of [['checkpoint_sha256',doc.meta.checkpoint_sha256],['policy_sha256',doc.meta.model_sha256],['simulator_sha256',doc.meta.simulator_sha256]])if(m[k]!==v)fail();
  for(const k of ['critic_sha256','state_sha256','network_sha256','counterfactual_sha256','reward_ledger_sha256','adapter_sha256','source_metrics_sha256'])if(!hash(m[k]))fail();
  if(!m.reward_hashes||!Object.keys(m.reward_hashes).length||Object.values(m.reward_hashes).some(v=>!hash(v))||!Array.isArray(m.component_names)||!m.component_names.length||new Set(m.component_names).size!==m.component_names.length)fail();
  if(!['original_trajectory_recomputed','same_checkpoint_new_seed','best_model_replacement'].includes(m.origin)||typeof m.selection_rule!=='string'||!m.selection_rule)fail();
  if(Object.keys(d).some(k=>!['schema','scale','legal_count',...FIELDS].includes(k))||[...FIELDS,'legal_count'].some(k=>!Array.isArray(d[k])||d[k].length!==n))fail();
  const rewardRows=new Map(doc.rewards.map(r=>[r[0],r]));
  for(let i=0;i<n;i++){
   for(const [rk,mk]of [['shaped_critic_q1_taken','shaped_q1_taken'],['shaped_critic_q1_policy_mean','shaped_q1_mean'],['shaped_advantage_a1','shaped_a1'],['shaped_advantage_delta','shaped_a1_increment']])if(doc.route?.[rk]?.[i]!==d[mk][i])fail();
   if(doc.route?.shaped_legal_count?.[i]!==d.legal_count[i])fail();
   const parts=rewardRows.get(i+1)?.[1]||{},extra=['control_entry_reward','control_residual_reward','control_clawback','route_shaped_reward'].reduce((s,k)=>s+(parts[k]||0),0);
   if(Math.abs((d.shaped_q1_taken[i]-d.raw_q1_taken[i])*d.scale-extra)>3e-5)fail();
   if(!Number.isInteger(d.legal_count[i])||d.legal_count[i]<1||d.legal_count[i]>15)fail();
   for(const k of FIELDS){if(k.endsWith('_increment')&&i===0){if(d[k][i]!==null)fail();}else if(!Number.isSafeInteger(d[k][i]))fail();}
   for(const role of ['raw','shaped'])if(Math.abs(d[role+'_a1'][i]-(d[role+'_q1_taken'][i]-d[role+'_q1_mean'][i]))>2)fail();
   if(Math.abs(d.delta_a[i]-(d.shaped_a1[i]-d.raw_a1[i]))>2)fail();
   for(const k of ['raw_a1','shaped_a1','delta_a'])if(i&&Math.abs(d[k+'_increment'][i]-(d[k][i]-d[k][i-1]))>2)fail();
  }
  for(const row of doc.rewards)for(const parts of row.slice(1))for(const [k,v]of Object.entries(parts))if(!m.component_names.includes(k)||!Number.isFinite(v))fail();
  if(doc.meta.reward_status!=='offline_recomputed_sealed_reward'||doc.meta.diagnostic_admission?.status!=='qualified')fail();
 }
 return {SCHEMA,FIELDS,atTick,validate};
});
