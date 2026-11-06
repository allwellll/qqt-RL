'use strict';
// Independent policy arithmetic repair; native coordinates/physics are untouched.
const V5=require('./interceptor_defend_cap4_bot_v5');
const VERSION='bun-interceptor-defend-cap4-bot-v6-f32-time-ties';
function halfUlp32(value){
 const a=Math.abs(value);
 return a===0||a<2**-126?2**-150:2**(Math.floor(Math.log2(a))-24);
}
function timeCeil(distance,distanceError,step,stepError){
 if(!(step>stepError&&step>0))return Math.ceil(distance/step-1e-9);
 const ratio=distance/step,nearest=Math.round(ratio);
 const low=Math.max(0,distance-distanceError)/(step+stepError);
 const high=(distance+distanceError)/(step-stepError);
 // A binary32 observation represents an interval of original values. Only
 // an integer inside that interval is treated as the deterministic ETA tie.
 // Genuine non-boundary fractional times retain the original ceiling.
 return nearest>=low&&nearest<=high?nearest:Math.ceil(ratio-1e-9);
}
class BunInterceptorBot extends V5.BunInterceptorBot{
 firstMove(player,a,optimistic){
  const factor=.3*(optimistic?1.25:1),step=factor*player.speed,stepError=factor*halfUlp32(player.speed);
  const vertical=a<2,cy=player.row+.5,cx=player.col+.5;
  const perpendicular=vertical?player.x:player.y,perpCenter=vertical?cx:cy;
  const pos=vertical?player.y:player.x,center=vertical?cy:cx,sign=a===0||a===2?-1:1;
  const perp=Math.abs(perpendicular-perpCenter),perpTicks=perp>.02?timeCeil(perp,halfUlp32(perpendicular),step,stepError):0;
  const along=Math.abs(center+sign-pos),leave=Math.max(0,sign*(center+sign*.45-pos));
  return {arrive:Math.max(1,perpTicks+timeCeil(along,halfUlp32(pos),step,stepError)),
   leave:perpTicks+timeCeil(leave,halfUlp32(pos),step,stepError)};
 }
 analyze(state,pid){const d=super.analyze(state,pid);return this.lastDecision={...d,action_schema:VERSION};}
}
module.exports={...V5,VERSION,ACTION_SCHEMA:Object.freeze({...V5.ACTION_SCHEMA,version:VERSION,
 policy_time_tie:'nearest integer only when contained in propagated binary32 half-ULP observation interval'}),
 BunInterceptorBot,halfUlp32,timeCeil};
