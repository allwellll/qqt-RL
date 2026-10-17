'use strict';
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.QQTReviewAdvantage=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const SCHEMA='qqt.policy-weighted-a1/v1';
  const COLUMNS=['ticks','taken_q','mean_q','advantage','delta'];
  // Candidates must already share the production state, opponent action and RNG snapshot.
  function calculate(candidates,taken,gamma){
    if(!Number.isFinite(gamma)||gamma<0||gamma>1)throw Error('gamma');
    const legal=candidates.filter(a=>a.legal);
    if(!legal.length||legal.some(a=>!Number.isFinite(a.probability)||a.probability<0||!Number.isFinite(a.reward)||!Number.isFinite(a.nextValue)))throw Error('legal candidates');
    const mass=legal.reduce((n,a)=>n+a.probability,0);
    const selected=legal.find(a=>a.action[0]===taken[0]&&a.action[1]===taken[1]);
    if(!selected||mass<=0)throw Error('taken action / policy mass');
    const q=a=>a.reward+gamma*(a.terminal?0:a.nextValue);
    const takenQ=q(selected),meanQ=legal.reduce((n,a)=>n+a.probability/mass*q(a),0);
    return {taken_q:takenQ,mean_q:meanQ,advantage:takenQ-meanQ};
  }
  function validate(doc){
    const meta=doc.meta.advantage,data=doc.advantages;
    if(meta===undefined&&data===undefined)return;
    const fail=()=>{throw Error('advantage schema / evidence');};
    if(!meta||meta.schema!==SCHEMA||!['recorded','not_recorded'].includes(meta.status)||!Number.isFinite(meta.gamma)||meta.gamma<0||meta.gamma>1||meta.action_scope!=='joint_move_ability'||typeof meta.legal_actions!=='string'||!meta.legal_actions||typeof meta.reason!=='string')fail();
    for(const k of ['checkpoint_sha256','policy_sha256','simulator_sha256'])if(meta[k]!==doc.meta[k==='policy_sha256'?'model_sha256':k])fail();
    if(meta.critic_sha256!==null&&!/^[a-f0-9]{64}$/.test(meta.critic_sha256))fail();
    if(!data||data.schema!==SCHEMA||Object.keys(data).some(k=>!['schema','scale','legal_count',...COLUMNS].includes(k))||!Number.isFinite(data.scale)||data.scale<=0||data.scale>1e-3)fail();
    if(data.legal_count!==undefined&&(!Array.isArray(data.legal_count)||data.legal_count.length!==data.ticks.length||data.legal_count.some(n=>!Number.isInteger(n)||n<1||n>15)))fail();
    if(COLUMNS.some(k=>!Array.isArray(data[k])||data[k].length!==data.ticks.length))fail();
    if(meta.status==='not_recorded'&&(data.ticks.length||meta.critic_sha256!==null||!meta.reason))fail();
    if(meta.status==='recorded'&&(!meta.critic_sha256||!data.ticks.length||!meta.counterfactual_sha256||!/^[a-f0-9]{64}$/.test(meta.counterfactual_sha256)))fail();
    if(meta.calculation_version&&(meta.measured_count!==data.ticks.length||meta.missing_count!==doc.actions.length-data.ticks.length||!data.legal_count||!meta.reward_hashes||!doc.meta.reward_hashes||JSON.stringify(Object.keys(meta.reward_hashes).sort())!==JSON.stringify(Object.keys(doc.meta.reward_hashes).sort())||Object.keys(meta.reward_hashes).some(k=>meta.reward_hashes[k]!==doc.meta.reward_hashes[k])))fail();
    for(let i=0;i<data.ticks.length;i++){
      const t=data.ticks[i],prior=data.ticks[i-1];
      if(!Number.isInteger(t)||t<1||t>doc.actions.length||(i&&t<=prior))fail();
      if(['taken_q','mean_q','advantage'].some(k=>!Number.isSafeInteger(data[k][i])))fail();
      if(Math.abs(data.advantage[i]-(data.taken_q[i]-data.mean_q[i]))>2)fail();
      const delta=data.delta[i];
      if(i===0||t!==prior+1){if(delta!==null)fail();}
      else if(!Number.isSafeInteger(delta)||Math.abs(delta-(data.advantage[i]-data.advantage[i-1]))>2)fail();
    }
  }
  function atTick(doc,tick){
    const data=doc.advantages,i=data?.ticks.indexOf(tick)??-1;
    return Object.fromEntries(['taken_q','mean_q','advantage','delta'].map(k=>[k,i<0||data[k][i]===null?null:data[k][i]*data.scale]));
  }
  function legalCount(doc,tick){const i=doc.advantages?.ticks.indexOf(tick)??-1;return i<0?null:doc.advantages.legal_count?.[i]??null;}
  function missing(meta){
    return {schema:SCHEMA,definition:'Q1=r+gamma*V(next); A1=Q1(taken)-sum(pi_legal*Q1)',
      status:'not_recorded',reason:'生产 Web 导出仅含 Actor；独立训练 Critic 未导出并验证，反事实 reward ledger 分支未验证',gamma:meta.gamma,
      checkpoint_sha256:meta.checkpoint_sha256,policy_sha256:meta.model_sha256,critic_sha256:null,simulator_sha256:meta.simulator_sha256,
      action_scope:'joint_move_ability',legal_actions:'production legalMask.mm × TransformerModel._abilityMask; joint probabilities renormalized over legal candidates',
      rng_status:'seeded simulator RNG restorable by draw count; counterfactual ledger equivalence not measured'};
  }
  const empty=()=>({schema:SCHEMA,scale:1e-6,ticks:[],taken_q:[],mean_q:[],advantage:[],delta:[]});
  return {SCHEMA,calculate,validate,atTick,legalCount,missing,empty};
});
