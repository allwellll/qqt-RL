'use strict';
// Two evidence-selected experimental changes only. This opt-in CPU candidate
// does not alter the frozen v1 pressure set, production default or training pool.
const I=require('../web/bun_interceptor_bot'),H=require('../web/bun_hunter_bot');
class BunInterceptorBot extends I.BunInterceptorBot{
 routePlan(...args){
  const plan=super.routePlan(...args);if(!plan)return plan;
  // A static ray detour treats the bomb as permanent. It does not establish
  // that the carrier will still be in the flame when this real fuse expires.
  const candidates=plan.candidates.filter(c=>c.timed);
  return {...plan,candidates,best:candidates[0]||null};
 }
 chooseGoal(state,g,pid,pred,perceive){
  this.interception=this.routePlan(state,g,pid);
  if(this.interception?.best)return {mode:'INTERCEPT',seeds:[{cell:this.interception.best.plant,cost:0}],threat:false,noDig:true};
  if(this.interception)return {mode:'INTERCEPT',seeds:super.attackSeeds(state,g,pid,pred,perceive),threat:false};
  return H.BunHunterBot.prototype.chooseGoal.call(this,state,g,pid,pred,perceive);
 }
 considerBomb(state,g,pid,pred,perceive,field){
  const plan=this.interception;
  if(!plan)return super.considerBomb(state,g,pid,pred,perceive,field);
  const me=state.players[pid];
  if(me.bombsLeft<=0||me.liveBombs>=this.cfg.maxLiveBombs||g.bombAt[me.cell]>=0)return null;
  if(plan.best?.plant===me.cell&&plan.best.plantAt>1)return null;
  const ray=I.footprint(state,g,me.cell,me.blast);
  const hits=plan.near.some(c=>{
   const t=plan.foe.slideTicks+Math.max(this.firstTicks(g,plan.foe,c,false),plan.from[c]*this.moveTicks(plan.foe,false));
   return ray[c]&&Math.abs(t-H.NEW_BOMB_TICK)<=this.moveTicks(plan.foe,false)+H.FLAME_TICKS;
  });
  if(!hits)return null;
  const mine={cell:me.cell,e:H.NEW_BOMB_TICK,blast:me.blast},withBomb=this.predict(state,g,[mine],perceive);
  const escape=this.escape(g,withBomb,me,false);let safe=escape.surv;
  if(this.cfg.robust)safe&=this.hypoSurvivors(state,g,pid,perceive,[mine]);
  for(const q of this.alliesOf(state,pid))if(state.players[q].alive&&!this.escape(g,withBomb,state.players[q],true).surv)safe=0;
  if(!safe)return null;
  return {move:this.pickMove(g,me,safe,escape,field,true,this.dangerCells(withBomb,g.N)),reason:'bomb_intercept_timed'};
 }
}
module.exports={BunInterceptorBot,VERSION:'bun-interceptor-timing-v2-experimental'};
