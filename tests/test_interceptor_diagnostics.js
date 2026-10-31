'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('../web/sim-bun-native-v2'),{Original,DiagnosticBot}=require('./interceptor_diagnostic_eval');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
let ticks=0;
for(const seed of [2026336000,2026336001]){
 const a=new Q.Sim(seed),b=new Q.Sim(seed),old=new Original({seed:73}),observed=new DiagnosticBot({seed:73});
 for(const s of [a,b]){s.reset(level,{nativeItems:true,nativeTrap:true,playerModes:['old','old']});s.maxSteps=160;}
 for(let t=0;t<160&&!a.done;t++){
  const action=old.act(a,1),same=observed.act(b,1);assert.deepEqual(same,action);
  const other=[t%5,t%17===0?1:0,0];a.step([other,action]);b.step([other,same]);
  for(const key of ['pos','fuse','alive','trapped','bunCarried','bunStored'])assert.deepEqual(b[key],a[key]);ticks++;
 }
}
console.log(JSON.stringify({status:'PASS',ticks,action_and_state_equal:true,scope:'passive diagnostic instrumentation'}));
