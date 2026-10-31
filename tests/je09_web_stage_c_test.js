'use strict';
const assert=require('node:assert/strict');
process.env.JE09_NAMESPACE='runs/je09_bun_20261015/stage_c_return_set_pfsp8_v8';
const stage=require('../scripts/je09_web_stage_b.js');
assert.equal(stage.manifest.seed,2041000000);assert.equal(stage.manifest.holdout_seed,2051000000);
const tracker=new stage.Tracker(0);tracker.returnPath=[{cell:1,distance:20},{cell:2,distance:19},{cell:1,distance:20},{cell:3,distance:19}];
const r=tracker.result();assert.equal(r.reverse_cell_transitions,1);assert.equal(r.unique_return_cells,3);assert.equal(r.downhill_cell_transitions,2);
for(let g=0;g<16;g++){const {s,team,enemy,initial}=stage.setup((stage.manifest.seed+g*104729)>>>0,g);assert.equal(s.bunCarried[team],-1);assert.equal(s.bunCarried[enemy],enemy);assert.ok(initial.distance>=2&&initial.distance<=8);}
console.log('Stage C independent seed manifest, true drop init and return-path stats passed');
