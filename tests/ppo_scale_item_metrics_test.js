'use strict';
const assert=require('node:assert/strict');
const Q=require('../web/sim'),P=require('../web/review_protocol');
const authority=require('../scripts/eval_web_hunter');
const {ItemMetrics}=require('../scripts/ppo_scale_item_metrics_v3');
function fresh(){const s=new Q.Sim(20261025);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
  for(const k of ['wall','brick','fuse','crate','pushable','blastLinger'])s[k].fill(0);
  s.pos.set([5.5,5.5,9.5,10.5]);s.invuln.fill(0);s.spawnProtection.fill(0);
  s.itemSlots[0]=[{item:Q.ITEM_BANANA,count:2}];s._syncHeldItem(0);return s;}
const a=fresh(),b=fresh(),m=new ItemMetrics(b,0);
for(let t=0;t<60;t++){const action=[t<8?3:2,0,Number(t===0)];
  m.before(action);a.step([action,[4,0,0]]);b.step([action,[4,0,0]]);m.after();
  assert.deepEqual(P.state(a),P.state(b));}
m.flush();assert.equal(m.chains.size,0);
assert.ok(m.chainEvents.length>0);
for(const e of m.chainEvents)if(e.owned_contact)assert.ok(e.preparation_ticks>=0);
// Full-chain cost includes placement-to-contact preparation and explicit terminal censoring.
const s=fresh(),n=new ItemMetrics(s,0);n.before([4,0,1]);s._placeHeldItem(0);n.after();
s.t+=12;n.flush();const e=n.chainEvents[0];assert.equal(e.censored,true);
assert.equal(e.ticks,12);assert.equal(n.row('combat',1).chain_beneficial,0);
console.log('PPO scale item v3: whole placement accounting, censor, no-physics-change passed');
