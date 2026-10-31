'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict'),{chromium}=require('/tmp/qqt-review-browser/node_modules/playwright');
(async()=>{
 const b=await chromium.launch({headless:true,args:['--no-sandbox']});const p=await b.newPage(),errors=[],writes=[];
 p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 await p.route('**/*',r=>{if(['GET','HEAD'].includes(r.request().method()))return r.continue();writes.push(r.request().method());return r.abort()});
 await p.goto(process.env.WEB_URL+'fixture.html');
 const result=await p.evaluate(async()=>{
  const d=await (await fetch('fixtures.json')).json(),m=await QQTModelWorker.load(d.row),rows=[];
  for(const f of d.fixtures){const r=await m.forward(Float32Array.from(f.obs),Float32Array.from(f.state));rows.push({tick:f.tick,seat:f.seat,max_error:Math.max(...['move','bomb'].flatMap(k=>Array.from(r[k],(v,i)=>Math.abs(v-f.expected[k][i]))),Math.abs(r.value-f.expected.value))});}
  const latency=[];const f=d.fixtures[1];for(let i=0;i<30;i++){const t=performance.now();await m.forward(Float32Array.from(f.obs),Float32Array.from(f.state));latency.push(performance.now()-t)}
  latency.sort((a,b)=>a-b);const result={backend:m.backend,policy_sha256:m.meta.checkpoint_sha256,encoding:m.meta.r3_input_encoding,state_dim:m.meta.state_dim,obs_shape:m.meta.obs_shape,zero_max_error:m.parityMaxError,rows,inference_p95_ms:latency[Math.floor(latency.length*.95)],inference_samples:30};m.close();return result;
 });
 assert.equal(result.backend,'worker-wasm');assert.equal(result.state_dim,37);assert(result.zero_max_error<1e-3);assert(result.rows.every(r=>r.max_error<.002));assert(result.inference_p95_ms<100);assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
 fs.writeFileSync('runs/v2_fresh_selection_20261031/R3_BROWSER_ABI.json',JSON.stringify({status:'PASS',...result,errors,writes},null,2));await b.close();console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1});
