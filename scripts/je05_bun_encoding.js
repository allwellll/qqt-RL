'use strict';
// Explicit native Bun v2 state encoding; dimensions/actions stay unchanged.
// Global9/10: dead=0, free alive=1, trapped alive=1+remainingTicks/60.
function encodeState(sim,pid){
  const vector=sim.encodeStateJAX(pid);
  vector[9]+=sim.trapped[pid]/60;vector[10]+=sim.trapped[1-pid]/60;
  return vector;
}
module.exports={encodeState,version:'bun24-trap-timer-v2'};
