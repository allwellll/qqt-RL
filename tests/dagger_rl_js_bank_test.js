'use strict';
const assert=require('node:assert/strict');
const Q=require('../web/sim');
const authority=require('../scripts/eval_web_hunter');
const P=require('../web/review_protocol');
const bank=require('../scripts/dagger_rl_js_bank');

for(const seed of [0,1,2026102581]){
  const expected=Q.mulberry32(seed),actual=bank.statefulRng(seed);
  for(let i=0;i<10000;i++)assert.equal(actual(),expected());
  const saved=actual.snapshot(),tail=Array.from({length:32},()=>actual());
  actual.restore(saved);assert.deepEqual(Array.from({length:32},()=>actual()),tail);
}
let physicsSteps=0;
for(const seed of [51,52]){
  const baseline=new Q.Sim(seed),instrumented=new bank.StatefulSim(seed);
  for(const s of [baseline,instrumented])s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
  assert.deepEqual(P.state(instrumented),P.state(baseline));
  for(let tick=0;tick<64;tick++){
    const clone=bank.restore(bank.capture(instrumented));
    const moves=[[tick%5,Number(tick%19===0),Number(tick%17===0)],[(tick+2)%5,Number(tick%23===0),0]];
    baseline.step(moves);instrumented.step(moves);clone.step(moves);physicsSteps+=3;
    assert.deepEqual(P.state(instrumented),P.state(baseline),'instrumentation changed production physics');
    assert.deepEqual(P.state(clone),P.state(instrumented),'complete native next step differs');
    assert.deepEqual(clone.rng.snapshot(),instrumented.rng.snapshot());
    assert.deepEqual(bank.capture(clone),bank.capture(instrumented),'hidden native state missing');
  }
}
const s=new bank.StatefulSim(7);s.unknownCallback=()=>0;
assert.throws(()=>bank.capture(s),/unrecognized function/);
assert.throws(()=>bank.restore({schema:'replay-only'}),/bank schema/);
// Review Player counts RNG draws by wrapping an already advanced stream.
const wrapped=new bank.StatefulSim(23);
wrapped.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
const expected=bank.restore(bank.capture(wrapped)),prior=wrapped.rng;let observedDraws=0;
wrapped.rng=()=>{observedDraws++;return prior();};
for(let i=0;i<64;i++)assert.equal(wrapped.rng(),expected.rng(),'Review wrapper reset the advanced RNG stream');
assert.equal(observedDraws,64);
assert.deepEqual(wrapped.rng.snapshot(),expected.rng.snapshot());
// Passive wrappers have external ownership/slide state; cloning physics alone
// would silently drop this auxiliary state in a real production trace.
const {ItemMetrics}=require('../scripts/dagger_rl_metrics_v2');
const native=new bank.StatefulSim(19);
native.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
const metric=new ItemMetrics(native,0);
metric.owned.set(31,{item:1,id:2,phase:'combat',cell:31});metric.serial=2;
for(let tick=0;tick<32;tick++){
  const saved=bank.captureInstrumented(native,metric);
  const {sim:clone,metrics:copy}=bank.restoreInstrumented(saved);
  assert.deepEqual(bank.captureInstrumented(clone,copy),saved);
  const action=[tick%5,Number(tick%17===0),Number(tick%13===0)];
  metric.before(action);copy.before(action);
  native.step([action,[4,0,0]]);clone.step([action,[4,0,0]]);
  physicsSteps+=2;
  metric.after();copy.after();
  assert.deepEqual(bank.captureInstrumented(clone,copy),bank.captureInstrumented(native,metric));
}
console.log(JSON.stringify({passed:true,physics_steps:physicsSteps,optimizer_steps:0,
  RNG_output_comparisons:30000,new_training_samples:0,scope:'technical native state clone tests, no synthetic carrier training'}));
