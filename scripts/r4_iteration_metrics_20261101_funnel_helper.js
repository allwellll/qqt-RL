'use strict';
// Passive transition ABI shared by Review recomputation and new JS evaluation.
// All state arguments are plain copies made immediately before/after Sim.step.
const snapshot=s=>({alive:Array.from(s.alive),trapped:Array.from(s.trapped),carry:Array.from(s.bunCarried),stored:s.bunStored.map(a=>a.slice()),pos:Array.from(s.pos)});
const active=(s,q)=>!s.alive[q]||s.trapped[q]>0;
function source(info,q,p,kind){const a=info.causalDamageSource?.[q]||[];return {kind,learner_causal_bomb:Boolean(a[p]),enemy_self_causal_bomb:Boolean(a[q]),attribution:a[p]&&!a[q]?'learner_only':a[q]&&!a[p]?'enemy_self_only':a[p]&&a[q]?'shared':'unknown'};}
class FunnelTracker{
 constructor(learner){this.p=learner;this.q=1-learner;this.windows=[];this.carries=[];this.events=[];this.window=null;this.carry=null;this.phase={combat:0,window:0,carrying:0};}
 observe(before,after,info,tick){
  const p=this.p,q=this.q,was=active(before,q),now=active(after,q);
  this.phase[before.carry[p]>=0?'carrying':was?'window':'combat']++;
  if(!was&&now){this.window={id:this.windows.length+1,start_tick:tick,end_tick:null,start_empty_hand:before.carry[p]<0,own_actionable:before.alive[p]&&!before.trapped[p],source:source(info,q,p,!after.alive[q]?'enemy_death':'native_sugar'),pickups:[],events:[]};this.windows.push(this.window);this.events.push({tick,type:'window_start',window_id:this.window.id,source:this.window.source});}
  if(this.window){if(before.alive[q]&&!after.alive[q])this.window.events.push({tick,type:'enemy_death'});if(!before.trapped[q]&&after.trapped[q])this.window.events.push({tick,type:'enemy_native_sugar'});}
  if(this.carry&&before.carry[p]>=0&&after.carry[p]!==before.carry[p]){
   const origin=before.carry[p],deposit=after.stored[p][origin]>before.stored[p][origin];Object.assign(this.carry,{end_tick:tick,end_reason:deposit?'delivery':info.died?.[p]?'death_drop':before.carry[q]<0&&after.carry[q]===origin?'observed_contact_transfer':'drop_or_transfer_unknown',delivered:deposit,active_enemy_origin:origin===q});this.events.push({tick,type:'carry_end',carry_id:this.carry.id,reason:this.carry.end_reason,origin});this.carry=null;
  }
  if(before.carry[p]<0&&after.carry[p]>=0||before.carry[p]>=0&&after.carry[p]>=0&&before.carry[p]!==after.carry[p]){
   const origin=after.carry[p],stock=before.stored[origin][origin]>after.stored[origin][origin],w=this.window&&(was||now)?this.window:null;
   this.carry={id:this.carries.length+1,start_tick:tick,end_tick:null,origin,active_enemy_origin:origin===q,pickup_type:stock?'stock':'loose_or_transfer',window_id:w?.id??null,pickup_during_window:Boolean(w),end_reason:null,delivered:false};this.carries.push(this.carry);if(w)w.pickups.push(this.carry.id);this.events.push({tick,type:'pickup',carry_id:this.carry.id,origin,pickup_type:this.carry.pickup_type,window_id:w?.id??null});
  }
  if(was&&!now&&this.window){this.window.end_tick=tick;this.window.end_reason=after.alive[q]?'enemy_recovered_or_respawned':'unknown';this.events.push({tick,type:'window_end',window_id:this.window.id});this.window=null;}
  if(info.died?.[p])this.events.push({tick,type:'learner_death',carried_before:before.carry[p],own_bomb_defeat:Boolean(info.ownBombDefeat?.[p]),was_controlled:before.trapped[p]>0});
 }
 finish(tick,maxTicks){
  if(this.window){Object.assign(this.window,{end_tick:tick,end_reason:'terminal_censored'});this.window=null;}
  if(this.carry){Object.assign(this.carry,{end_tick:tick,end_reason:tick>=maxTicks?'time_limit_carried':'terminal_carried'});this.carry=null;}
  for(const w of this.windows){w.duration_ticks=w.end_tick-w.start_tick;w.stock_pickups=this.carries.filter(c=>c.window_id===w.id&&c.pickup_type==='stock'&&c.active_enemy_origin).length;w.same_carry_enemy_origin_deliveries=this.carries.filter(c=>c.window_id===w.id&&c.active_enemy_origin&&c.delivered).length;}
  const cs=this.carries,ws=this.windows;
  return {version:'r4-observable-funnel-v1',phase:this.phase,windows:ws,carries:cs,events:this.events,funnel:{windows:ws.length,empty_hand_window_starts:ws.filter(w=>w.start_empty_hand).length,learner_only_created_windows:ws.filter(w=>w.source.attribution==='learner_only').length,unknown_created_windows:ws.filter(w=>w.source.attribution==='unknown').length,window_enemy_stock_pickups:cs.filter(c=>c.active_enemy_origin&&c.pickup_type==='stock'&&c.pickup_during_window).length,window_same_carry_deliveries:cs.filter(c=>c.active_enemy_origin&&c.pickup_during_window&&c.delivered).length,enemy_stock_pickups:cs.filter(c=>c.active_enemy_origin&&c.pickup_type==='stock').length,enemy_origin_deliveries:cs.filter(c=>c.active_enemy_origin&&c.delivered).length,death_drops:cs.filter(c=>c.end_reason==='death_drop').length,unknown_drops:cs.filter(c=>c.end_reason==='drop_or_transfer_unknown').length,terminal_carried:cs.filter(c=>c.end_reason?.includes('carried')).length},limitations:['same-origin within-tick drop and reacquisition is unobservable from step-boundary states','causalDamageSource attributes bomb window-start only; unknown not guessed','window entry is observational, not forced intervention benefit']};
 }
}
module.exports={snapshot,active,source,FunnelTracker};
