'use strict';
const assert=require('node:assert/strict'),b=require('../scripts/je09_web_stage_b.js'),p=require('../scripts/je08_web_diagnostics.js');
for(const base of [b.manifest.seed,b.manifest.holdout_seed])for(let g=0;g<16;g++){
 const {s,team,enemy,initial}=b.setup((base+g*104729)>>>0,g);
 assert.ok(initial.distance>=2&&initial.distance<=8);assert.ok(initial.return_distance<=24);assert.ok(initial.return_distance>initial.enemy_distance);
 assert.equal(s.bunCarried[team],-1);assert.equal(s.bunCarried[enemy],enemy);
 const t=new b.Tracker(team);t.trapCause[enemy][team]=true;const before=p.snapshot(s);t.observe(s,before,s.step([[4,0,0],[4,0,0]]));
 assert.equal(t.result().eligible_drop,true);assert.equal(t.result().first_pickup,false);
}
const t=new b.Tracker(0);Object.assign(t,{first:10,startDistance:12,returnTick:20,marks:[true,true,true],captureTick:90,broken:false});
t.trace=[{event:'pickup_return',tick:20,distance:11},{event:'progress25',tick:30,distance:9},{event:'progress50',tick:150,distance:6},{event:'home_half',tick:40,distance:8}];
Object.assign(t,{firstHome:false});assert.equal(t.result().progress25,true);assert.equal(t.result().progress50,false);assert.equal(t.result().home_half,true);
t.firstHome=true;assert.equal(t.result().home_half,false);
console.log('Stage B real-drop setup and finite-window strict crossing passed');
