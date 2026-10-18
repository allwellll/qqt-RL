'use strict';
// Observational ledger; never mutates production state or actions.
class ControlFunnel {
 constructor(team=0){this.team=team;this.owner=-1;this.event=0;this.metrics={first_control:0,attributed_first_control:0,final_pop:0,active_contact_pop:0,causal_kill:0,surviving_kill:0,native_causal_kill:0};}
 observe(before,after,info){
  const a=this.team,v=1-a,m=this.metrics;
  if(before.trapped[v]===0&&after.trapped[v]>0){
   m.first_control++;this.event++;
   const src=info.causalDamageSource?.[v]||[];
   this.owner=src.filter(Boolean).length===1&&src[a]&&!src[v]?a:-1;
   m.attributed_first_control+=Number(this.owner===a);
  }
  const pop=before.trapped[v]>0&&info.died[v];
  const causal=pop?this.owner===a:Boolean(info.causalKill?.[a]);
  m.final_pop+=Number(pop);
  // Remaining >1 excludes passive expiry; expiry-tick contact is conservatively unknown.
  m.active_contact_pop+=Number(pop&&before.trapped[v]>1);
  m.native_causal_kill+=Number(info.causalKill?.[a]);
  m.causal_kill+=Number(causal);
  m.surviving_kill+=Number(causal&&after.alive[a]&&!info.died[a]);
  if(info.died[v]||(before.trapped[v]>0&&after.trapped[v]===0))this.owner=-1;
  return m;
 }
}
module.exports={ControlFunnel};
