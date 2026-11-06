'use strict';
// Rolling front/side interception. Every request obeys the current native cap.
// Finite public-coordinate rollouts are a heuristic, never an all-path proof.
const Base=require('./interceptor_complete_v8_p8');
const Front=require('./interceptor_front_cutoff_p0_v2');
const Route=require('../web/bun_interceptor_bot');
const Motion=require('./interceptor_public_attribute_witness_v1');
const Q=require('../web/sim-bun-native-v2');
const VERSION='bun-interceptor-defend-cap4-bot-v2';
const ACTION_SCHEMA=Object.freeze({...Base.ACTION_SCHEMA,version:VERSION,memory:['noProgress'],control:'finite public-coordinate rollout; no exhaustive kill certificate'});
function popcount(n){let k=0;for(;n;n&=n-1)k++;return k;}
function localCover(state,pid,pred){
 const p=state.players[pid],g={H:13,W:15,N:195},pos=[p.y,p.x];
 // Current contact and its actual four adjacent exit contacts; one simultaneous
 // impact must cover every contact, never a union across distinct flame ticks.
 const ends=[pos,[pos[0]-.9,pos[1]],[pos[0]+.9,pos[1]],[pos[0],pos[1]-.9],[pos[0],pos[1]+.9]];
 let best=0;
 for(let t=Math.max(1,p.invuln+1);t<=31;t++){
  const c=pred.impact.subarray(t*195,(t+1)*195);let mask=0;
  for(let k=0;k<ends.length;k++)if(ends[k][0]>=0&&ends[k][0]<13&&ends[k][1]>=0&&ends[k][1]<15&&Motion.hit(ends[k],c))mask|=1<<k;
  best=Math.max(best,popcount(mask));
 }
 return best;
}
class BunInterceptorBot extends Base.BunInterceptorBot{
 analyze(state,pid){
  const baseline=super.analyze(state,pid),me=state.players[pid],foe=state.players[1-pid],g=this.geometry(state);
  const carrying=foe.alive&&foe.trapped<=0&&foe.carrying===(me.team??pid);
  let action=baseline.action,reason=baseline.reason,target=-1,bomb=false,before=0,after=0,gain=0,safe=0,gate=0,coverGain=0;
  let active=false;
  if(carrying&&me.alive&&me.trapped<=0&&me.carrying<0&&me.movementStatus!==2&&!baseline.witness_used){
   const pred=Front.nativePrediction(this.predict(state,g,[],()=>true));
   const escape=this.escape(g,pred,me,false);
   const home=Route.distances(g,this.baseCells(state,g,foe.team??(1-pid))),from=Route.distances(g,[foe.cell]);
   const blocked=!!this.noProgress?.blocked;
   const plan=blocked?null:Front.cutoffPlan(state,pid,this,pred);
   // A stalled carrier invalidates the home-route timing assumption. Approach
   // the real carrier instead of bombing a distant unused return corridor.
   const near=from[me.cell]<=4;
   target=(!near&&plan?.best)?plan.best.plant:foe.cell;
   const field=Route.distances(g,[target]);
   active=!!escape.surv;
   if(active){
    let move=Front.selectMove(g,me,escape.surv,field,pred);
    gate=blocked?5:near?6:2;
    const capacity=me.bombsLeft>0&&me.liveBombs<me.bombsCap&&g.bombAt[me.cell]<0&&!state.brick[me.cell];
    if(capacity){
     const pb=Front.nativePrediction(this.predict(state,g,[{cell:me.cell,e:31,blast:me.blast}],()=>true));
     const own=this.escape(g,pb,me,false);safe=own.surv;
     if(safe){
      const rb=Front.nativeRoutes(state,1-pid,pred,this),ra=Front.nativeRoutes(state,1-pid,pb,this),common=rb.supportedMask&ra.supportedMask;
      const b=rb.mask&common,a=ra.mask&common;before=popcount(b);after=popcount(a);gain=popcount(b&~a);
      coverGain=localCover(state,1-pid,pb)-localCover(state,1-pid,pred);
      // Close coordinated crossfire can add a side exit even when the nine
      // selected controls retain survivors. Timing uses chain-resolved impact.
      const inFoot=pb.impact[31*195+foe.cell]&&!pred.impact[31*195+foe.cell];
      const closeCut=near&&coverGain>0&&(inFoot||coverGain>=2);
      const forwardCut=!blocked&&plan?.best&&me.cell===plan.best.plant&&gain>0;
      if(closeCut||forwardCut){
       bomb=true;gate=1;
       // Continue toward a side of the carrier while respecting own escape;
       // do not return to the just-planted appointment and wait out the fuse.
       move=Front.selectMove(g,me,safe,from,pb);
      }
     }
    }
    action=[move,bomb?1:0];reason=bomb?'defend_rolling_crossfire':blocked?'defend_stalled_carrier_pressure':near?'defend_close_side':'defend_rolling_front';
    // Preserve the full native baseline's active pop/control and immediate
    // survival item decisions. There is no fabricated new item heuristic.
    if(baseline.action[1]>=2||baseline.reason==='bomb_kill'){action=baseline.action;reason=baseline.reason;bomb=false;gate=7;}
   }
  }
  return this.lastDecision={...baseline,action,reason,action_schema:VERSION,
   front_cutoff_active:active,front_cutoff_target:target,front_cutoff_plant_at:-1,
   front_cutoff_bomb:bomb,front_cutoff_carrier_paths_before:before,front_cutoff_carrier_paths_after:after,
   front_cutoff_control_gain:gain,front_cutoff_placement_gate:gate,front_cutoff_safe_moves:safe,
   front_cutoff_predicted_kill:false,defend_contact_exit_gain:coverGain,defend_capacity:me.bombsCap,defend_live_bombs:me.liveBombs};
 }
}
module.exports={...Base,VERSION,ACTION_SCHEMA,BunInterceptorBot,localCover};
