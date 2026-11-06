'use strict';
// Observational protection, never claims an unobserved counterfactual rescue.
function summarize(source,r3,p){
 const rows=(source.custodies||[]).filter(x=>x.player===p).map(c=>{
  const contacts=(r3.contacts||[]).filter(e=>e.tick>=c.start_tick&&e.tick<=c.end_tick);
  const banana=contacts.filter(e=>e.player===p&&e.owner===p&&e.type===1);
  const glue=contacts.filter(e=>e.player!==p&&e.owner===p&&e.type===2);
  const protection=banana.length&&glue.length?'banana_and_glue':banana.length?'banana_self_contact':glue.length?'glue_enemy_contact':'no_observed_protection';
  return {...c,protection,banana_contacts:banana.length,glue_contacts:glue.length};
 });
 const groups={};
 for(const name of ['all','banana_and_glue','banana_self_contact','glue_enemy_contact','no_observed_protection']){
  const a=name==='all'?rows:rows.filter(c=>c.protection===name),delivered=a.filter(c=>c.outcome==='same_custody_delivery').length;
  groups[name]={acquisitions:a.length,same_chain_deliveries:delivered,delivery_rate:a.length?delivered/a.length:null,carrying_deaths:a.filter(c=>c.outcome==='death_drop').length,timeouts:a.filter(c=>c.outcome==='timeout_still_carrying').length,controlled_drops:a.filter(c=>c.outcome==='controlled_drop').length,unknown_drops:a.filter(c=>c.outcome==='drop_transfer_unknown').length,other_terminal:a.filter(c=>c.outcome==='other_terminal_still_carrying').length};
 }
 return {version:'interceptor-primary-custody-protection-v1',learner:p,groups,custodies:rows,limitations:['contact-based protection exposure is observational, not causal prevention','unobserved within-tick custody changes remain unknown']};
}
module.exports={summarize};
