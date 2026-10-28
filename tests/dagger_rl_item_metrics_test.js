'use strict';
const assert=require('node:assert/strict');
const Q=require('../web/sim.js'),authority=require('../scripts/eval_web_hunter.js');
const {ItemMetrics}=require('../scripts/dagger_rl_metrics_v2');
function fresh(){const s=new Q.Sim(20261024);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
 for(const k of ['wall','brick','fuse','crate','pushable','blastLinger'])s[k].fill(0);
 s.pos.set([5.5,5.5,9.5,10.5]);s.invuln.fill(0);s.spawnProtection.fill(0);s.itemSlots[0]=[{item:Q.ITEM_BANANA,count:2}];s._syncHeldItem(0);return s;}
for(const requested of [undefined,null,-1,5*Q.W+5]){
 const s=fresh(),m=new ItemMetrics(s,0),cell=5*Q.W+5;
 m.before([4,0,1]);assert.equal(s._placeHeldItem(0,requested),true);m.after();
 assert.equal(s.fieldOwner[cell],0);assert.equal(m.rows.combat_banana.legal_releases,1);
 s.pos[1]=6.5;s._armFieldItems();s.pos[1]=5.5;
 m.before([3,0,0]);s._contactFieldItem(0,3);m.after();
 assert.equal(s.movementStatus[0],Q.MOVE_STATUS_SLIDE);
 assert.equal(m.rows.combat_banana.owned_self_contacts,1,'real own contact must resolve native fallback placement cell');
 assert.equal(m.rows.combat_banana.other_self_contacts,0);
 m.before([4,0,0]);m.after();assert.equal(m.rows.combat_banana.owned_self_contacts,1,'no duplicate contact');
}
// A passive correction must preserve every production physics/RNG state.
const {ItemMetrics:LegacyMetrics}=require('../scripts/dagger_rl_metrics');
const P=require('../web/review_protocol.js');
const a=fresh(),b=fresh(),old=new LegacyMetrics(a,0),fixed=new ItemMetrics(b,0);
for(let tick=0;tick<60;tick++){
 const action=[tick<8?3:2,0,tick===0?1:0];
 old.before(action);fixed.before(action);
 a.step([action,[4,0,0]]);b.step([action,[4,0,0]]);
 old.after();fixed.after();
 assert.deepEqual(P.state(a),P.state(b),'metric correction must leave physics and RNG exact');
}
console.log('native fallback-cell ownership tests passed; mechanism fixtures only, no training');
