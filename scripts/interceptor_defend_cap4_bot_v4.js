'use strict';
// Native bubble occupancy intercepts the carrier before fuse expiry. Adjacent
// placements can chain on one tick; inventory and own escape remain binding.
const V2=require('./interceptor_defend_cap4_bot_v2'),Route=require('../web/bun_interceptor_bot'),F=require('./interceptor_front_cutoff_p0_v2');
const VERSION='bun-interceptor-defend-cap4-bot-v4';
function blockingPlan(state,pid,bot,g){
 const me=state.players[pid],foe=state.players[1-pid],home=Route.distances(g,bot.baseCells(state,g,foe.team??1-pid)),from=Route.distances(g,[foe.cell]),own=Route.distances(g,[me.cell]),short=home[foe.cell],rows=[];
 for(let c=0;c<195;c++){
  if(!g.open[c]||g.bombAt[c]>=0||home[c]>=short||from[c]<1||from[c]>8||from[c]+home[c]>short+1)continue;
  const eta=from[c]*bot.moveTicks(foe,false),arrival=c===me.cell?0:Math.max(bot.firstTicks(g,me,c,false),own[c]*bot.moveTicks(me,false));
  if(arrival+1>eta)continue;
  const blocked=new Uint8Array(195);blocked[c]=1;const detour=Route.distances(g,bot.baseCells(state,g,foe.team??1-pid),blocked)[foe.cell]-short;
  const exits=[0,1,2,3].reduce((n,a)=>n+Number(g.nb[c*4+a]>=0&&g.open[g.nb[c*4+a]]),0);
  // Prefer an actual cut (including finite detour) rather than raw bubble count.
  if(detour<2&&exits>2)continue;
  rows.push({plant:c,arrival,eta,detour,exits,score:arrival+from[c]*2+(exits>2?8:0)-(detour>=1000000?12:Math.min(detour,8))});
 }
 rows.sort((a,b)=>a.score-b.score||a.plant-b.plant);return rows[0]||null;
}
class BunInterceptorBot extends V2.BunInterceptorBot{
 analyze(state,pid){
  const d=super.analyze(state,pid),me=state.players[pid],foe=state.players[1-pid];let blocker=-1,detour=0;
  const carrying=foe.alive&&foe.trapped<=0&&foe.carrying===(me.team??pid);
  if(carrying&&d.front_cutoff_active&&!this.noProgress?.blocked&&d.action[1]===0&&d.reason!=='bomb_kill'){
   const g=this.geometry(state),pred=F.nativePrediction(this.predict(state,g,[],()=>true)),esc=this.escape(g,pred,me,false),p=blockingPlan(state,pid,this,g);
   if(p&&esc.surv){
    blocker=p.plant;detour=p.detour;const field=Route.distances(g,[p.plant]);let move=F.selectMove(g,me,esc.surv,field,pred),bomb=false,safe=0;
    if(me.cell===p.plant&&me.bombsLeft>0&&me.liveBombs<me.bombsCap&&g.bombAt[me.cell]<0){
     const pb=F.nativePrediction(this.predict(state,g,[{cell:me.cell,e:31,blast:me.blast}],()=>true));safe=this.escape(g,pb,me,false).surv;
     if(safe){bomb=true;move=F.selectMove(g,me,safe,Route.distances(g,[foe.cell]),pb);}
    }
    d.action=[move,bomb?1:0];d.reason=bomb?'defend_native_body_block_bomb':'defend_native_body_block_front';d.front_cutoff_target=p.plant;d.front_cutoff_bomb=bomb;d.front_cutoff_placement_gate=bomb?1:2;d.front_cutoff_safe_moves=safe;
   }
  }
  return this.lastDecision={...d,action_schema:VERSION,defend_blocker_target:blocker,defend_blocker_detour:detour};
 }
}
module.exports={...V2,VERSION,ACTION_SCHEMA:Object.freeze({...V2.ACTION_SCHEMA,version:VERSION}),BunInterceptorBot,blockingPlan};
