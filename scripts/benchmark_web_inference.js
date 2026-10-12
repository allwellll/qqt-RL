'use strict';
// Deterministic production models/states, real Chromium + real Worker/WASM.
const fs=require('fs'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try {
    const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(process.env.WEB_URL||'http://127.0.0.1:8094/');
    await page.waitForFunction(()=>window.QQTModelWorker && window.QQTModelCatalog);
    const result=await page.evaluate(async()=>{
      const manifest=await(await fetch('models.json')).json();
      const summary=values=>{const sorted=values.slice().sort((a,b)=>a-b);return {n:values.length,min:sorted[0],median:sorted[Math.floor(sorted.length/2)],p95:sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.95))],max:sorted.at(-1),mean:values.reduce((a,b)=>a+b,0)/values.length};};
      const load=async id=>{
        const row=manifest.models.find(r=>r.id===id);const buffer=await(await fetch('models/'+row.file)).arrayBuffer();
        await QQTModelLoader.verifyBuffer(buffer,row);
        return {row,doc:JSON.parse(new TextDecoder().decode(buffer))};
      };
      const old=await load('bun-tf-it4000'),current=await load('bun392-selfplay3072');
      const levels=await(await fetch('assets/maps/levels.json')).json(),level=levels.find(l=>l.qqt_id===806);
      const samples=[];
      for(let seed=0;seed<8;seed++){
        const sim=new QQT.Sim(3920400+seed);sim.reset(level,{nativeItems:seed%2===1,nativeTrap:seed%2===1});
        // Native item and carry states without changing any encoder contract.
        if(seed%2){sim.heldItem[0]=2;sim.heldItem[1]=1;}
        for(let t=0;t<seed;t++)sim.step([[t%5,0,0],[(t+2)%5,0,0]]);
        for(let pid=0;pid<2;pid++)samples.push({sim,pid,obs:sim.encodeObsJAX(pid,24),state:sim.encodeStateJAX(pid)});
      }
      const timeModel=model=>{const times=[];for(let warm=0;warm<3;warm++)model.forward(samples[0].obs,samples[0].state);for(const s of samples){const t=performance.now();model.forward(s.obs,s.state);times.push(performance.now()-t);}return summary(times);};
      const oldJs=timeModel(new QQT.TransformerModel(old.doc));
      const js=new QQT.TransformerModel(current.doc);js.greedy=true;
      const currentJs=timeModel(js);
      const worker=await QQTModelWorker.load(current.row);worker.greedy=true;
      const maxErrors=[];let actionMismatches=0;
      const workerTimes=[];let frames=[],last=performance.now(),running=false;const longs=[];
      const observer=new PerformanceObserver(list=>longs.push(...list.getEntries().map(e=>e.duration)));
      const raf=now=>{frames.push(now-last);last=now;if(running)requestAnimationFrame(raf);};
      const expected=samples.map(s=>js.forward(s.obs,s.state));
      await new Promise(r=>setTimeout(r,100));
      frames=[];longs.length=0;last=performance.now();running=true;requestAnimationFrame(raf);observer.observe({type:'longtask',buffered:false});
      for(let iteration=0;iteration<3;iteration++)for(let i=0;i<samples.length;i++){
        const s=samples[i],t=performance.now();const actual=await worker.forward(s.obs,s.state);workerTimes.push(performance.now()-t);
        maxErrors.push(Math.max(...['move','bomb'].flatMap(k=>Array.from(actual[k],(v,j)=>Math.abs(v-expected[i][k][j]))),Math.abs(actual.value-expected[i].value)));
        const mask=s.sim.legalMask();
        const decode=f=>js._decodeAbility(js._sampleMasked(f.move,mask.mm[s.pid],QQT.mulberry32(7)),js._sampleMasked(f.bomb,js._abilityMask(s.sim,mask,s.pid),QQT.mulberry32(7)));
        if(JSON.stringify(decode(actual))!==JSON.stringify(decode(expected[i])))actionMismatches++;
      }
      await new Promise(r=>setTimeout(r,80));running=false;observer.disconnect();
      // Full deterministic native trajectories include masks, action decoding,
      // simulator state/RNG and carried items, not just isolated logits.
      let trajectoryMismatches=0;
      for(let seed=0;seed<2;seed++) {
        const left=new QQT.Sim(3923072+seed),right=new QQT.Sim(3923072+seed);
        for(const sim of [left,right]) {sim.reset(level,{nativeItems:true,nativeTrap:true});sim.heldItem[0]=2;}
        const lrng=QQT.mulberry32(512+seed),rrng=QQT.mulberry32(512+seed);
        for(let tick=0;tick<16;tick++) {
          const a=js.act(left,0,lrng),b=await worker.act(right,0,rrng);
          if(JSON.stringify(a)!==JSON.stringify(b))trajectoryMismatches++;
          const opponent=[tick%5,0,0];left.step([a,opponent]);right.step([b,opponent]);
          const ls=Array.from(left.encodeObsJAX(0,24)),rs=Array.from(right.encodeObsJAX(0,24));
          if(JSON.stringify(ls)!==JSON.stringify(rs))trajectoryMismatches++;
        }
      }
      worker.close();
      const adapter=navigator.gpu?await navigator.gpu.requestAdapter():null;
      return {webgpu_adapter_available:!!adapter,old_js_ms:oldJs,bun392_js_ms:currentJs,bun392_worker_ms:summary(workerTimes),worker_backend:worker.backend,
        backend_reason:worker.backendReason||null,initial_parity_max_error:worker.parityMaxError,
        max_logit_value_error:Math.max(...maxErrors),greedy_action_mismatches:actionMismatches,trajectory_ticks:32,trajectory_mismatches:trajectoryMismatches,
        frame_ms:summary(frames.filter(v=>v>0)),main_long_tasks_ms:longs,webgpu_available:!!navigator.gpu,
        input_contract:{obs:[24,13,15],global:24,ability:3},model_sha256:current.row.sha256,onnx_sha256:current.row.onnx.sha256};
    });
    console.log(JSON.stringify(result,null,2));
    fs.writeFileSync(process.env.EVIDENCE_FILE||'/tmp/qqt-web-inference-benchmark.json',JSON.stringify(result,null,2)+'\n');
    assert.equal(result.worker_backend,'worker-wasm');assert.equal(result.greedy_action_mismatches,0);assert.equal(result.trajectory_mismatches,0);assert.ok(result.max_logit_value_error<1e-3);
    assert.ok(result.bun392_worker_ms.p95<100);assert.ok(result.main_long_tasks_ms.every(v=>v<100));assert.deepEqual(errors,[]);
    fs.writeFileSync(process.env.EVIDENCE_FILE||'/tmp/qqt-web-inference-benchmark.json',JSON.stringify({passed:true,...result},null,2)+'\n');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
