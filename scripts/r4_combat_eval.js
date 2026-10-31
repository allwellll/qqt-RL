'use strict';
// Isolated evaluation scene; no production rule or old opponent mutation.
const Q=require('../web/sim-bun-native-v2');
if(process.env.R4_COMBAT_SCENE==='clear'){
 const reset=Q.Sim.prototype.reset;
 Q.Sim.prototype.reset=function(...args){const r=reset.apply(this,args);this.brick.fill(0);this.pushable.fill(0);this.pushBoxAt.fill(-1);this.pushBoxes=[];this._gen++;return r;};
}
require('./r4_combat_metrics').install(Q);
require('./r3_items_candidate_eval');
