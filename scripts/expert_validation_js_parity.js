'use strict';
// C-side fixture: fixed map 806, fixed initial state/actions, no event RNG.
// The paired Python script emits the JAX trace; this writes the production-JS
// trace using the same action sequence and state fields.
const fs=require('node:fs');
const crypto=require('node:crypto');
const Q=require('../web/sim.js');
const E=require('./eval_web_hunter.js');
const seed=2026102860, ticks=48;
const level=E.loadLevel();
const sim=new Q.Sim(seed);sim.reset(level,{nativeTrap:true,nativeItems:true});
sim.pos.set([4.5,2.5,4.5,12.5]);
sim.bunSpawnPos=[[4.5,2.5],[4.5,12.5]];
sim.alive=[true,true];sim.hp=[1,1];sim.invuln=[0,0];sim.spawnProtection=[0,0];
sim.trapped=[0,0];sim.bunCarried=[-1,-1];sim.bunStored=[[1,0],[0,1]];
sim.fuse.fill(0);sim.owner.fill(-1);sim.bombBlast.fill(0);sim.blastLinger.fill(0);
sim.itemSlots=[[],[]];sim.heldItem=[0,0];sim.fieldItem.fill(0);sim.fieldOwner.fill(-1);
sim._gen+=1;
const moves=Array.from({length:ticks},(_,t)=>[[t%8<4?3:2,0,0],[t%8<4?2:3,0,0]]);
const hash=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
const row=()=>{const m=sim.legalMask();return {t:sim.t,pos:Array.from(sim.pos),alive:sim.alive.slice(),trapped:sim.trapped.slice(),
  bun_carried:sim.bunCarried.slice(),bun_stored:sim.bunStored.map(x=>x.slice()),brick_hash:hash(Array.from(sim.brick)),wall_hash:hash(Array.from(sim.wall)),
  crate_hash:hash(Array.from(sim.crate)),fuse_hash:hash(Array.from(sim.fuse)),field_item_hash:hash(Array.from(sim.fieldItem)),held_item:sim.heldItem.slice(),move_status:sim.movementStatus.slice(),move_mask:m.mm,ability_mask:m.am,
  observation:[sim.encodeObsJAX(0,24),sim.encodeObsJAX(1,24)].map(x=>Array.from(x))};};
const trace=[row()];
for(const a of moves){sim.step(a);trace.push(row());}
const out={schema:'qqt.expert-validation-js-parity/v1',engine:'production JS web/sim.js',seed,ticks,actions:moves,initial:trace[0],trace,
  source_sha256:{'web/sim.js':hash(fs.readFileSync('web/sim.js')),'web/assets/maps/levels.json':hash(fs.readFileSync('web/assets/maps/levels.json'))}};
const output=process.argv[2]||'runs/expert_validation_20261025/C_js_trace.json';fs.mkdirSync(require('node:path').dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(out));
console.log(output);
