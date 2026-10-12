'use strict';
const fs=require('fs'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  try {
    const context=await browser.newContext();const errors=[],writes=[];
    await context.route('**/*',route=>{
      const r=route.request(),pathname=new URL(r.url()).pathname;
      if(['GET','HEAD'].includes(r.method())||(r.method()==='POST'&&/\/rpc\/qqt_(leaderboard|get_profile|my_win_leaderboard)$/.test(pathname)))return route.continue();
      writes.push(pathname);return route.abort();
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript(()=>{
      const interval=window.setInterval;
      window.tickAttempts=0;window.tickCompletions=0;window.tickLatency=[];
      window.setInterval=(callback,ms,...args)=>interval(async()=>{
        if(ms===100){const t=performance.now();tickAttempts++;await callback(...args);tickCompletions++;tickLatency.push(performance.now()-t);}
        else callback(...args);
      },ms);
    });
    await page.goto(process.env.WEB_URL||'http://127.0.0.1:8094/');
    await page.waitForFunction(()=>window.QQTModelWorker && document.querySelector('#model-options option[value="model:bun392-selfplay3072"]'));
    await page.evaluate(()=>{
      const load=QQTModelWorker.load;QQTModelWorker.load=async(...args)=>{const m=await load(...args);window.performanceModel=m;return m;};
      const reset=QQT.Sim.prototype.reset;QQT.Sim.prototype.reset=function(...args){window.liveSim=this;return reset.apply(this,args);};
    });
    await page.click('#advanced-open');await page.fill('#settings-password','demaxiya');await page.click('#password-form button[type="submit"]');
    await page.selectOption('#opponent','model:bun392-selfplay3072');
    await page.waitForFunction(()=>document.querySelector('#model-status').textContent.startsWith('已加载：Bun392动态自对弈3072'),null,{timeout:180000});
    const modes=[];
    for(const mode of ['human-vs-opponent','model-vs-rule']){
      await page.selectOption('#match-mode',mode);await page.waitForTimeout(500);
      const r=await page.evaluate(async()=>{
        const sim=liveSim,start=sim.t,attempt=tickAttempts,complete=tickCompletions,li=tickLatency.length,mi=performanceModel.inferenceTimes.length;
        const frames=[],longs=[];let previous=performance.now(),running=true;
        const observer=new PerformanceObserver(list=>longs.push(...list.getEntries().map(e=>e.duration)));observer.observe({type:'longtask',buffered:false});
        const raf=now=>{frames.push(now-previous);previous=now;if(running)requestAnimationFrame(raf);};requestAnimationFrame(raf);
        const startTime=performance.now();await new Promise(r=>setTimeout(r,10000));const wall=performance.now()-startTime;
        running=false;observer.disconnect();
        const stats=x=>{const a=x.slice().sort((a,b)=>a-b);return {n:x.length,p95:a[Math.floor(a.length*.95)],max:a.at(-1),mean:x.reduce((a,b)=>a+b,0)/x.length};};
        return {ticks:sim.t-start,wall_ms:wall,tick_attempts:tickAttempts-attempt,tick_completions:tickCompletions-complete,
          tick_latency_ms:stats(tickLatency.slice(li)),inference_ms:stats(performanceModel.inferenceTimes.slice(mi)),
          frame_ms:stats(frames.filter(x=>x>0)),long_tasks_ms:longs,backend:performanceModel.backend,
          native_bun:sim.isBun,native_items:sim.nativeItems,native_trap:sim.nativeTrap,players:sim.nPlayers};
      });
      console.log(JSON.stringify({mode,...r}));
      assert.equal(r.backend,'worker-wasm');assert(r.native_bun);assert.equal(r.native_items,mode==='human-vs-opponent');assert.equal(r.native_trap,mode==='human-vs-opponent');assert.equal(r.players,2);
      assert(r.ticks>=98 && r.ticks>=r.tick_attempts-1,JSON.stringify(r));assert(r.inference_ms.p95<100);assert(r.long_tasks_ms.every(ms=>ms<100));
      modes.push({mode,...r});
    }
    assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
    const result={passed:true,network_mock:false,modes,errors,writes};
    fs.writeFileSync(process.env.EVIDENCE_FILE||'/tmp/qqt-bun-inference-browser.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
