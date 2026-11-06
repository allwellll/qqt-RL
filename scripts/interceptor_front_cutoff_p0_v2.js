'use strict';
// Independent opt-in front policy; frozen v1/v8 and production rules stay intact.
const Base=require('./interceptor_complete_v8_p8');
const V1=require('./interceptor_front_cutoff_p0_v1');
const Route=require('../web/bun_interceptor_bot');
const Motion=require('./interceptor_public_attribute_witness_v1');
const Q=require('../web/sim-bun-native-v2');
const VERSION='bun-interceptor-front-cutoff-p0-v2';
const ACTION_SCHEMA=Object.freeze({...Base.ACTION_SCHEMA,version:VERSION,
 placement:'pre-move current cell; selected target and due planned tick',
 control:'finite native-coordinate routes; sample reduction is not an exhaustive capture certificate'});
function nativePrediction(pred){return {...pred,lethal:pred.impact};}
// Nine deterministic public-state continuations: five constant controls and four
// shortest-home policies with rotated direction ties. No future policy or RNG.
function nativeRoutes(state,pid,pred,bot,horizon=34){
 const g=bot.geometry(state),me=state.players[pid],goals=bot.baseCells(state,g,me.team??pid);
 const home=Route.distances(g,goals),paths=[];
 for(let route=0;route<9;route++){
  let pos=[me.y,me.x],status=me.movementStatus,speed=me.speed,hit=false,unknown=status===3,delivered=false;
  const trace=[];
  const routeMove=(at,t)=>{
   const c=Math.max(0,Math.min(194,Math.floor(at[0])*15+Math.floor(at[1])));
   let move=route<5?route:4;
   if(route>=5){let best=home[c];for(let k=0;k<4;k++){const a=(k+route-5)%4,n=g.nb[c*4+a];
    if(n>=0&&g.open[n]&&pred.bombGone[n]<=t&&home[n]<best){move=a;best=home[n];}}}
   return move;
  };
  for(let t=1;t<=horizon;t++){
   const cell=Math.max(0,Math.min(194,Math.floor(pos[0])*15+Math.floor(pos[1])));
   const move=routeMove(pos,t);
   const blocked=Uint8Array.from(state.wall,(v,c)=>{
    const naturalFuse=state.bombs.find(b=>b.cell===c)?.fuse??31;
    const bombBlocked=pred.bombGone[c]>t||(pred.bombGone[c]===t&&naturalFuse>t);
    return !!(v||(state.brick[c]&&(!pred.brickGone[c]||t<pred.brickGone[c]))||bombBlocked);
   });
   const movement=at=>{
    if(status!==2)return Motion.movement(at,routeMove(at,t),blocked,speed);
    const p=Q.Sim.prototype._tryMove.call({},at[0],at[1],me.slideDir,blocked,.3*speed);
    return [Math.min(Math.max(p[0],Q.CFG.radius),13-Q.CFG.radius),Math.min(Math.max(p[1],Q.CFG.radius),15-Q.CFG.radius)];
   };
   // Host JS uses binary64 scratch. The JAX twin promotes only this finite
   // kinematic calculation; physical inputs and production state stay float32.
   const next=movement(pos),covered=pred.impact.subarray(t*195,(t+1)*195);
   if(status===2&&Math.abs(next[0]-pos[0])+Math.abs(next[1]-pos[1])<=2e-4){status=0;speed=me.postSlideSpeed||.6;}
   const end=Math.max(0,Math.min(194,Math.floor(next[0])*15+Math.floor(next[1])));
   // Known current carried/glue/slide speeds are used. Future status/pickup
   // changes are explicitly unknown, never turned into a false capture.
   unknown ||= !!state.fieldItem[cell]||!!state.fieldItem[end]||!!state.knownThreatWitness?.pushable?.[end];
   const struck=t>me.invuln&&Motion.hit(next,covered);
   hit ||= struck;
   delivered ||= goals.includes(end);
   trace.push({tick:t,move:status===2?me.slideDir:move,pos:next.slice(),contact:Motion.contactCells(next),hit:struck,unknown});
   pos=next;if(hit||delivered)break;
  }
  paths.push({route,alive:!hit,supported:!unknown,delivered,trace});
 }
 const mask=paths.reduce((m,p)=>m|(!p.hit&&p.alive&&p.supported?1<<p.route:0),0);
 const supportedMask=paths.reduce((m,p)=>m|(p.supported?1<<p.route:0),0);
 return {mask,supportedMask,count:paths.filter(p=>p.alive&&p.supported).length,paths};
}
class BunInterceptorBot extends Base.BunInterceptorBot{
 analyze(state,pid){
  const baseline=super.analyze(state,pid),me=state.players[pid],g=this.geometry(state);
  const pred=nativePrediction(this.predict(state,g,[],()=>true)),p=V1.cutoffPlan(state,pid,this,pred);
  const esc=me.alive&&me.trapped<=0?this.escape(g,pred,me,false):{surv:0};
  const active=!!p?.best&&!!esc.surv&&me.movementStatus!==2&&baseline.reason!=='bomb_kill'&&!baseline.witness_used;
  let action=baseline.action,bomb=false,gain=0,safe=0,before=0,after=0,routeGain=0,gate=0;
  if(active){
   const field=Route.distances(g,[p.best.plant]);let move=V1.selectMove(g,me,esc.surv,field,pred);
   if(me.cell===p.best.plant&&p.best.plantAt>1&&(esc.surv&16))move=4;
   const capacity=me.carrying<0&&me.bombsLeft>0&&me.liveBombs<me.bombsCap&&g.bombAt[me.cell]<0&&!state.brick[me.cell];
   gate=me.cell!==p.best.plant?2:p.best.plantAt>1?3:0;
   if(capacity&&gate===0){
    const pb=nativePrediction(this.predict(state,g,[{cell:me.cell,e:31,blast:me.blast}],()=>true));
    const own=this.escape(g,pb,me,false);safe=own.surv;
    const rb=nativeRoutes(state,1-pid,pred,this),ra=nativeRoutes(state,1-pid,pb,this);
    const common=rb.supportedMask&ra.supportedMask;
    const validBefore=rb.mask&common,validAfter=ra.mask&common;
    before=bitCount(validBefore);after=bitCount(validAfter);
    routeGain=bitCount(validBefore&~validAfter);
    for(const c of p.near)for(let t=Math.max(1,p.eta[c]-2);t<=Math.min(64,p.eta[c]+2);t++)
     if(pb.impact[t*195+c]&&!pred.impact[t*195+c]){gain++;break;}
    if(routeGain>0&&safe){bomb=true;move=V1.selectMove(g,me,safe,field,pb);gate=1;}
    else gate=4;
   }
   action=[move,bomb?1:0];
  }
  return this.lastDecision={...baseline,action,action_schema:VERSION,
   reason:active?(bomb?'front_cutoff_timed_bomb':'front_cutoff'):baseline.reason,
   front_cutoff_active:active,front_cutoff_target:p?.best?.plant??-1,
   front_cutoff_carrier_eta:p?.best?.carrierETA??-1,front_cutoff_interceptor_eta:p?.best?.arrival??-1,
   front_cutoff_plant_at:p?.best?.plantAt??-1,front_cutoff_gain:p?.best?.gain??0,
   front_cutoff_bomb:bomb,front_cutoff_new_flame_cells:gain,front_cutoff_safe_moves:safe,
   front_cutoff_carrier_paths_before:before,front_cutoff_carrier_paths_after:after,
   front_cutoff_control_gain:routeGain,front_cutoff_placement_gate:gate,
   front_cutoff_predicted_kill:false,front_cutoff_slide_endpoint:p?.foe?.cell??-1};
 }
}
function bitCount(v){let n=0;for(;v;v&=v-1)n++;return n;}
module.exports={...Base,VERSION,ACTION_SCHEMA,cutoffPlan:V1.cutoffPlan,selectMove:V1.selectMove,
 nativePrediction,nativeRoutes,BunInterceptorBot};
