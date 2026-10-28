'use strict';
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.QQTReviewEvaluation=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  // Separate scope: reward-only evaluation never satisfies complete Critic admission.
  const SCHEMA='qqt.review-reward-only/v1';
  function validate(doc,model){
    const m=doc.meta,r=m.review_reward,fail=()=>{throw Error('reward-only evaluation evidence');};
    if(!model||m.model_id!==model.id||m.checkpoint_sha256!==model.checkpoint_sha256||m.model_sha256!==model.model_sha256||
       !r||r.schema!==SCHEMA||r.measured_count!==doc.actions.length||!r.scope||!r.limitations||
       m.reward_status!=='offline_recomputed_review_reward'||m.training_reward_status!=='unrecorded'||
       m.advantage?.status!=='not_recorded'||doc.advantages?.ticks.length!==0||
       doc.metrics!==undefined||doc.diagnostic_ticks!==undefined||doc.meta.diagnostic_admission!==undefined||
       doc.meta.approximate_a1!==undefined||doc.meta.route!==undefined||
       !Array.isArray(r.component_names)||!r.component_names.length||
       JSON.stringify(r.component_names)!==JSON.stringify(m.reward_components)||
       !m.reward_hashes||!Object.keys(m.reward_hashes).length||
       Object.values(m.reward_hashes).some(h=>!/^[a-f0-9]{64}$/.test(h))||
       !['win','loss','draw'].includes(doc.summary.outcome))fail();
    for(const row of doc.rewards)for(const parts of row.slice(1))for(const k of Object.keys(parts))if(!r.component_names.includes(k))fail();
    if(m.evaluation_setup==='best_d_clear_duel_v1'&&(m.native_rules!==false||m.scene!=='open'||m.max_ticks!==300))fail();
    return true;
  }
  return {SCHEMA,validate};
});
