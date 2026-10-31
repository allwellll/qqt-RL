'use strict';
// Diagnostic local movement hold only. Never installed by training or webpage.
function holdController(period){
 if(!Number.isInteger(period)||period<1)throw Error('positive integer period');
 const episodes=new WeakMap();let overrides=0,maskedHeld=0,decisions=0;
 return {apply(s,pid,action){let e=episodes.get(s);if(!e||s.t===0){e={move:action[0],lastTick:-1};episodes.set(s,e);}
  if(s.t%period===0||e.lastTick<0)e.move=action[0];
  e.lastTick=s.t;const result=action.slice();result[0]=e.move;decisions++;overrides+=result[0]!==action[0];
  maskedHeld+=Number(!s.legalMask().mm[pid][result[0]]);return result;},stats(){return {period,overrides,masked_held_ticks:maskedHeld,decision_ticks:decisions};}};
}
module.exports={holdController};
