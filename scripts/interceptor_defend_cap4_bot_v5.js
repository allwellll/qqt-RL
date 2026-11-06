'use strict';
// Observation-only numeric contract. Reuse frozen v4; never alter native Sim.
const V4=require('./interceptor_defend_cap4_bot_v4');
const VERSION='bun-interceptor-defend-cap4-bot-v5-float32-observation';
function float32Observation(state){
 const players=state.players.map(p=>{
  const y=Math.fround(p.y),x=Math.fround(p.x);
  const row=Math.max(0,Math.min(state.height-1,Math.floor(y))),col=Math.max(0,Math.min(state.width-1,Math.floor(x)));
  return {...p,y,x,row,col,cell:row*state.width+col,speed:Math.fround(p.speed),postSlideSpeed:Math.fround(p.postSlideSpeed??.6)};
 });
 const m=state.knownThreatWitness;
 // speedStep is an exact rule-signature constant in JS eligibility, not an
 // observed physical quantity. Preserve it, including unsupported signatures.
 const knownThreatWitness=m?{...m,growth:Array.from(m.growth||[],Math.fround),caps:Array.from(m.caps||[],Math.fround),crateRate:Math.fround(m.crateRate)}:m;
 return {...state,players,knownThreatWitness};
}
class BunInterceptorBot extends V4.BunInterceptorBot{
 analyze(state,pid){const d=super.analyze(float32Observation(state),pid);return this.lastDecision={...d,action_schema:VERSION};}
}
module.exports={...V4,VERSION,ACTION_SCHEMA:Object.freeze({...V4.ACTION_SCHEMA,version:VERSION,numeric_observation:'float32 public kinematics/attributes; exact JS rule signature constant'}),BunInterceptorBot,float32Observation};
