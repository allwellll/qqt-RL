'use strict';
// Opt-in composition over frozen v8. No simulator or collector is modified.
const Base=require('./interceptor_complete_v8_p8');
const Route=require('../web/bun_interceptor_bot');
const {NEW_BOMB_TICK:FUSE,FLAME_TICKS}=require('../web/bun_hunter_bot');
const VERSION='bun-interceptor-front-cutoff-p0-v1';
const ACTION_SCHEMA=Object.freeze({...Base.ACTION_SCHEMA,version:VERSION,
 request:['move','ability'],memory:['noProgress'],
 frontier:'strictly lower open-graph home distance than projected carrier',
 placement:'at most one native bomb request per logical tick; actual inventory cap'});
const INF=1000000;
function cutoffPlan(state,pid,bot=new Base.BunInterceptorBot({allowFallback:true}),prediction=null){
 const g=bot.geometry(state),me=state.players[pid],foe=state.players[1-pid];
 if(!me.alive||me.trapped>0||me.carrying>=0||!foe.alive||foe.trapped>0||foe.carrying!==(me.team??pid))return null;
 // Call the frozen geometric planner directly, without v4's no-progress gate.
 // It projects public banana motion and uses actual carried/glued speed.
 const p=Route.BunInterceptorBot.prototype.routePlan.call(bot,state,g,pid);
 if(!p)return null;
 const pred=prediction||bot.predict(state,g,[],()=>true),f=p.foe;
 const home=Route.distances(g,p.goals),own=Route.distances(g,[me.cell]);
 p.near=p.near.filter(c=>home[c]<p.short);
 p.common=p.common.filter(c=>p.near.includes(c));
 const eta=p.from.map((d,c)=>f.slideTicks+Math.max(bot.firstTicks(g,f,c,false),d*bot.moveTicks(f,false)));
 const existing=new Uint8Array(g.N);
 for(const c of p.near)if(eta[c]<=64&&pred.lethal[eta[c]*g.N+c])existing[c]=1;
 const candidates=[];
 for(let plant=0;plant<g.N;plant++){
  if(!g.open[plant]||g.bombAt[plant]>=0||own[plant]>=INF)continue;
  const ray=Route.footprint(state,g,plant,me.blast),hit=p.near.filter(x=>ray[x]);
  if(!hit.length)continue;
  const choke=hit.filter(x=>p.common.includes(x));
  const impact=(choke.length?choke:hit).sort((a,b)=>p.from[a]-p.from[b]||a-b)[0];
  const arrival=own[plant]===0?0:Math.max(bot.firstTicks(g,me,plant,false),own[plant]*bot.moveTicks(me,false));
  const carrierETA=eta[impact],timed=arrival+FUSE<=carrierETA+2,plantAt=Math.max(arrival,carrierETA-FUSE);
  if(!timed||carrierETA+3<arrival)continue;
  const gain=p.near.filter(x=>ray[x]&&!existing[x]&&Math.abs(eta[x]-(plantAt+FUSE))<=bot.moveTicks(f,false)+FLAME_TICKS).length;
  if(gain>0)candidates.push({plant,impact,arrival,carrierETA,plantAt,timed,common:choke.length>0,gain,
   score:(choke.length?0:10000)+plantAt+arrival-2*gain});
 }
 candidates.sort((a,b)=>a.score-b.score||a.plant-b.plant);
 return {...p,candidates,best:candidates[0]||null,eta,existing,
  target:candidates[0]?.plant??-1,carrierSpeed:f.speed,interceptorSpeed:me.speed};
}
function selectMove(g,me,mask,distance,pred){
 let move=4,best=Infinity;
 for(let a=0;a<5;a++){
  if(!(mask&(1<<a)))continue;
  const c=a===4?me.cell:g.nb[me.cell*4+a];if(c<0)continue;
  let danger=0;for(let t=1;t<=pred.T;t++)if(pred.lethal[t*g.N+c]){danger=1;break;}
  const score=Math.min(distance[c],999)*10+danger*80+(a===4?3:0);
  if(score<best){best=score;move=a;}
 }
 return move;
}
class BunInterceptorBot extends Base.BunInterceptorBot{
 analyze(state,pid){
  const baseline=super.analyze(state,pid),me=state.players[pid],foe=state.players[1-pid],g=this.geometry(state);
  const pred=this.predict(state,g,[],()=>true),p=cutoffPlan(state,pid,this,pred);
  const esc=me.alive&&me.trapped<=0?this.escape(g,pred,me,false):{surv:0};
  // Preserve native survival/witness/status handling and bounded kill attacks.
  // Ordinary close pressure must not keep pulling the candidate off its gate.
  const active=!!p?.best&&!!esc.surv&&me.movementStatus!==2&&baseline.reason!=='bomb_kill'&&!baseline.witness_used;
  let action=baseline.action,reason=baseline.reason,bomb=false,gain=0,before=0,after=0,safe=0;
  if(active){
   const distance=Route.distances(g,[p.best.plant]);
   let move=selectMove(g,me,esc.surv,distance,pred);
   if(me.cell===p.best.plant&&p.best.plantAt>1&&(esc.surv&16))move=4;
   const capacity=me.carrying<0&&me.bombsLeft>0&&me.liveBombs<me.bombsCap&&g.bombAt[me.cell]<0&&!state.brick[me.cell];
   if(capacity){
    const pb=this.predict(state,g,[{cell:me.cell,e:FUSE,blast:me.blast}],()=>true);
    // The predictor resolves real chain timing before testing the ETA window.
    for(const c of p.near){const lo=Math.max(1,p.eta[c]-2),hi=Math.min(64,p.eta[c]+2);let added=false;
     for(let t=lo;t<=hi;t++)if(t>foe.invuln&&pb.lethal[t*g.N+c]&&!pred.lethal[t*g.N+c])added=true;
     if(added)gain++;
    }
    const own=this.escape(g,pb,me,false);safe=own.surv;
    before=this.escape(g,pred,foe,true,FUSE+FLAME_TICKS).endCount;
    after=this.escape(g,pb,foe,true,FUSE+FLAME_TICKS).endCount;
    if(gain>0){
     // A lower reachable count is control, zero is a bounded kill prediction.
     // Forward flame gain is also useful even if total endpoint count is equal.
     if(safe){bomb=true;move=selectMove(g,me,safe,distance,pb);}
    }
   }
   action=[move,bomb?1:0];reason=bomb?'front_cutoff_timed_bomb':'front_cutoff';
  }
  return this.lastDecision={...baseline,action,reason,action_schema:VERSION,
   front_cutoff_active:active,front_cutoff_target:p?.best?.plant??-1,
   front_cutoff_carrier_eta:p?.best?.carrierETA??-1,front_cutoff_interceptor_eta:p?.best?.arrival??-1,
   front_cutoff_plant_at:p?.best?.plantAt??-1,front_cutoff_gain:p?.best?.gain??0,
   front_cutoff_bomb:bomb,front_cutoff_new_flame_cells:gain,front_cutoff_safe_moves:safe,
   front_cutoff_carrier_paths_before:before,front_cutoff_carrier_paths_after:after,
   front_cutoff_predicted_kill:bomb&&after===0,front_cutoff_slide_endpoint:p?.foe?.cell??-1};
 }
}
module.exports={...Base,VERSION,ACTION_SCHEMA,cutoffPlan,selectMove,BunInterceptorBot};
