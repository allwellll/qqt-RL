'use strict';
// Independent curriculum evaluation: regional first position, native own-side
// respawns. No changes to physical rules, rewards or action selection.
const fs=require('node:fs'),Q=require('../web/sim-bun-native-v2');
const spawn=JSON.parse(fs.readFileSync(process.argv[6],'utf8'));
if(spawn.schema!=='multi-region-eval-starts-v1')throw Error('explicit region manifest required');
const step=Q.Sim.prototype.step;
Q.Sim.prototype.step=function(actions){
 if(this.t===0){const i=(this.seed-spawn.seed)%spawn.spawn_cells.length;
  const pair=spawn.respawn_cells[i];if(!pair)throw Error('missing frozen native respawn');
  this.bunSpawnPos=pair.map(x=>[x[0]+.5,x[1]+.5]);}
 return step.call(this,actions);
};
require('./r4_combat_eval');
