'use strict';
const assert=require('node:assert/strict'),a=require('../scripts/je09_web_stage_a.js'),prev=require('../scripts/je08_web_diagnostics.js'),bridge=require('../scripts/je09_web_bridge.js');
const seeds=new Set();const starts=new Set(),dirs=new Set();
for(const base of [a.manifest.seed,a.manifest.holdout_seed])for(let g=0;g<16;g++){
 const seed=(base+g*104729)>>>0;assert.ok(!seeds.has(seed));seeds.add(seed);
 const {s,team,enemy,initial}=a.setup(seed,g);assert.ok(initial.distance>=2&&initial.distance<=8);assert.equal(s.bunCarried[team],-1);
 starts.add(initial.start);dirs.add(initial.orientation[team]);
 const t=new bridge.Tracker(team);t.trapCause[enemy][team]=true;const b=prev.snapshot(s);t.observe(s,b,s.step([[4,0,0],[4,0,0]]));
 assert.equal(t.result().eligible_drop,true);assert.equal(t.result().first_pickup,false);
}
assert.ok(starts.size>8);assert.equal(dirs.size,4);assert.equal(a.manifest.assist,false);console.log('stageA randomized spawn/orientation native causal drop test passed');
