'use strict';
// Independent P8 adapter. Complete-v4 remains the frozen baseline.
const V4=require('./interceptor_complete_v4');
const W=require('./interceptor_public_attribute_witness_v1');
const VERSION='bun-interceptor-complete-v7-p8-public-witness';

function stateFromSim(sim){
 const state=V4.stateFromSim(sim);
 state.knownThreatWitness=W.metadata(sim);
 return state;
}

function publicInput(state){
 return {...V4.publicInput(state),tick:state.tick,witness:state.knownThreatWitness};
}

class BunInterceptorBot extends V4.BunInterceptorBot{
 act(sim,pid=1){return V4.engineAction(this.analyze(stateFromSim(sim),pid));}
 analyze(state,pid){
  const base=super.analyze(state,pid);
  if(base.reason!=='doomed_max_survival')return {...base,witness_used:false,witness_reason:'baseline_not_doomed'};
  const witness=W.witness(state,pid);
  if(!witness.applicable||!witness.certifiedMask)return {...base,witness_used:false,witness_reason:witness.reason||'no_certified_escape'};
  return this.lastDecision={...base,action:[witness.selectedMove,0],reason:'public_witness_escape',
   witness_used:true,witness_reason:'certified',witness_mask:witness.certifiedMask,witness_horizon:witness.horizon};
 }
}

module.exports={...V4,VERSION,BunInterceptorBot,stateFromSim,publicInput};
