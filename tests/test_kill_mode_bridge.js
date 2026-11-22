'use strict';
const assert=require('node:assert/strict');
const {World,create,reward}=require('../scripts/kill_mode_bridge');
const input=require('../web/r3_model_inputs');
const seeds=[72,93],world=new World(seeds),reference=seeds.map(create);
for(let t=0;t<75;t++) {
  const actions=seeds.map((_,i)=>[[t%5,Number(t%11===i)],[4,Number(t%13===i)]]);
  const actual=world.step(actions);
  const raw=Buffer.from(actual.obs,'base64'),state=Buffer.from(actual.state,'base64');
  for(let i=0;i<seeds.length;i++) {
    const before={trapped:reference[i].trapped.slice()};
    const info=reference[i].step(actions[i].map(([m,a])=>[m,Number(a===1),Number(a===2)]));
    assert.deepEqual(world.sims[i].snapshotReplay(),reference[i].snapshotReplay(),'native physical state parity');
    assert.deepEqual(actual.rewards[i],reward(before,reference[i],info));
    for(let p=0;p<2;p++) {
      const x=input.inputs(reference[i],p),offset=(i*2+p)*25*195*4;
      const expected=Buffer.from(x.obs.buffer,x.obs.byteOffset,x.obs.byteLength);
      assert(raw.subarray(offset,offset+expected.length).equals(expected),'exact web observation ABI');
      const pos=(i*2+p)*38*4;
      assert(state.subarray(pos,pos+37*4).equals(Buffer.from(x.state.buffer)),'exact web state ABI');
      assert.equal(state.readFloatLE(pos+37*4),0,'mode row matches web export');
    }
  }
}
const dead=new World([1]);dead.sims[0]._killPlayer(0);
let observation=dead.observe();assert.deepEqual(observation.mm[0][0],[0,0,0,0,1]);
assert.deepEqual(observation.bm[0][0],[1,0,0]);
for(let t=0;t<30;t++)dead.step([[[4,0],[4,0]]]);
assert(dead.sims[0].alive[0]);assert.equal(dead.sims[0].invuln[0],10);
const terminal=new World([2]);terminal.sims[0].t=1799;
observation=terminal.step([[[4,0],[4,0]]]);assert.equal(observation.done[0],1);
assert.equal(terminal.sims[0].t,0);assert.equal(terminal.sims[0].killCount[0],0);
assert.throws(()=>world.step([]),/batch length/);
const fake={trapped:[0,0],done:false,winner:null};
const info={physicalDamageSource:[[false,false],[true,false]],creditedKill:[true,false],died:[false,true],ownBombDefeat:[false,false]};
assert.deepEqual(reward({trapped:[0,0]},fake,info),[5,-5]);
console.log('PASS kill bridge: 75-step native state/observation parity, death/respawn, terminal reset, factual reward');
