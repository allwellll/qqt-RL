'use strict';
// Require the passive adapter with a stub evaluator, and compare a deterministic
// native game against an uninstrumented process via subprocess (RNG unchanged).
const cp=require('node:child_process'),assert=require('node:assert/strict');
const script=String.raw`
const Q=require('./web/sim-stock-window-v1'),fs=require('fs'),path=require('path');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
if(process.env.INSTRUMENT==='1'){
 const filename=require.resolve('./scripts/bomb_threat_eval');require.cache[filename]={id:filename,filename,loaded:true,exports:{}};
 process.argv[4]='/tmp/r1-passive-parity-'+process.pid;
 require('./scripts/bomb_threat_r1_eval');
}
const s=new Q.Sim(level,{seed:123,nativeItems:true,nativeTrap:true});s.__btIdentity={seed:123,seat:0};
for(let t=0;t<120&&!s.done;t++)s.step([[t%5,Number(t%13===0)],[t%5,Number(t%19===0)]]);
const value={t:s.t,pos:[...s.pos],fuse:[...s.fuse],alive:s.alive,bunStored:s.bunStored};
console.log(JSON.stringify(value));
`;
const a=cp.execFileSync('node',['-e',script],{env:{...process.env,INSTRUMENT:'0'}}).toString();
const b=cp.execFileSync('node',['-e',script],{env:{...process.env,INSTRUMENT:'1'}}).toString();
assert.deepEqual(JSON.parse(a),JSON.parse(b));console.log('R1 passive native Sim state parity PASS');
