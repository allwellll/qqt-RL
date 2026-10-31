'use strict';
const assert=require('node:assert/strict');
const Q=require('../web/sim'),P=require('../web/review_protocol');
const authority=require('../scripts/eval_web_hunter');
const {ItemMetrics}=require('../scripts/ppo_scale_item_metrics_v5');
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
// Mechanism fixture: a real native inventory pickup precedes the whole slide.
// It is never used as a training source or a natural-game capability result.
const c=fresh();
for(let row=0;row<13;row++)c.wall[row*15]=c.wall[row*15+14]=1;
for(let col=0;col<15;col++)c.wall[col]=c.wall[180+col]=1;
c.wall[2*15+4]=1;
c.pos.set([2.5,11.5,10.5,12.5]);c.bunInitial[1]=2;c.bunStored[1][1]=2;
for(let t=0;t<9&&c.bunCarried[0]<0;t++)c.step([[2,0,0],[4,0,0]]);
assert.equal(c.bunCarried[0],1,'cargo must originate in an actual native pickup');
const full=new ItemMetrics(c,0);
for(let t=0;t<55;t++){const action=[t<5?3:2,0,Number(t===0)];
  full.before(action);c.step([action,[4,0,0]]);full.after();}
assert.equal(full.chainEvents.length,1);
const settled=full.chainEvents[0];
assert.equal(settled.censored,false);assert.equal(settled.owned_contact,true);
assert.equal(settled.delivered_during_chain,true);
assert.ok(settled.preparation_ticks>0);assert.ok(settled.net_home_distance>0);
assert.equal(full.row('carry',1).chain_beneficial,1);
// A second owned contact must explicitly censor, rather than overwrite, the first.
const o=fresh(),overlap=new ItemMetrics(o,0),cell=5*Q.W+5;
for(let k=0;k<2;k++){
  overlap.before([3,0,1]);assert.equal(o._placeHeldItem(0,cell+k),true);
  o.fieldArmed[cell+k]=1;o._contactFieldItem(0,3,cell+k);overlap.after();
}
assert.equal(overlap.chainEvents.length,1);
assert.equal(overlap.chainEvents[0].reason,'overlapping_contact');
assert.equal(overlap.chainEvents[0].censored,true);
o.done=true;overlap.before([4,0,0]);overlap.after();
assert.equal(overlap.chains.size,0);assert.equal(overlap.chainEvents.length,2);
assert.equal(overlap.chainEvents[1].reason,'terminal_or_horizon');
assert.notEqual(overlap.chainEvents[0].id,overlap.chainEvents[1].id);
// Placement alone cannot overwrite a previously consumed banana's active slide.
const z=fresh(),reuse=new ItemMetrics(z,0),at=5*Q.W+5;
reuse.before([3,0,1]);z._placeHeldItem(0,at);z.fieldArmed[at]=1;
z._contactFieldItem(0,3,at);reuse.after();
reuse.before([3,0,1]);z._placeHeldItem(0,at);reuse.after();
assert.equal(reuse.chainEvents.length,0,'preserve the first slide until actual end/contact overlap');
assert.equal(reuse.chains.size,2);
assert.equal(reuse.chains.get(2).preexistingMovementEffect,true);
// Even positive raw displacement cannot certify a newly placed item's benefit
// if it started under another item's motion effect. This is a mechanism fixture.
const w=fresh();
for(let row=0;row<13;row++)w.wall[row*15]=w.wall[row*15+14]=1;
for(let col=0;col<15;col++)w.wall[col]=w.wall[180+col]=1;
w.wall[2*15+4]=1;w.pos.set([2.5,11.5,10.5,12.5]);w.bunInitial[1]=2;w.bunStored[1][1]=2;
for(let t=0;t<9&&w.bunCarried[0]<0;t++)w.step([[2,0,0],[4,0,0]]);
assert.equal(w.bunCarried[0],1);w._setMovementStatus(0,Q.MOVE_STATUS_FAST,1);
const confounded=new ItemMetrics(w,0);
for(let t=0;t<55;t++){const action=[t<5?3:2,0,Number(t===0)];
  confounded.before(action);w.step([action,[4,0,0]]);confounded.after();}
assert.equal(confounded.chainEvents[0].preexisting_movement_effect,true);
assert.equal(confounded.row('carry',1).chain_beneficial,0);
console.log('PPO scale item v5: preparation, real pickup, delivery latch, overlap, terminal, parity passed');
