'use strict';
// Real catalog, real model download and native 1v1 actions; no result submission.
const assert=require('node:assert/strict'),fs=require('fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base=process.env.WEB_URL || 'https://allwellll.github.io/qqt-RL/';
const output=process.env.EVIDENCE_FILE || 'runs/bun-release-browser.json';
const id=process.env.MODEL_ID || 'bun392-selfplay3072';
const loadTimeout=Number(process.env.MODEL_LOAD_TIMEOUT_MS || 20*60*1000);
const http1Only=process.env.BROWSER_HTTP1_ONLY==='1';
assert.ok(Number.isFinite(loadTimeout) && loadTimeout>=30000 && loadTimeout<=60*60*1000);
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,
    args:['--no-sandbox',...(http1Only?['--disable-http2','--disable-quic']:[])]});
  const errors=[],writes=[],downloads=[],consoleErrors=[],runtimeWarnings=[],resourceWarnings=[],httpErrors=[];
  try {
    const context=await browser.newContext({viewport:{width:1440,height:1000}});
    context.on('request',request=>{
      const pathname=new URL(request.url()).pathname;
      if(['GET','HEAD'].includes(request.method()) ||
          (request.method()==='POST' && /\/rpc\/qqt_(leaderboard|get_profile|my_win_leaderboard)$/.test(pathname))) return;
      writes.push(pathname);
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{
      if(m.type()!=='error')return;
      // ORT writes this host CPU-detection warning to stderr even when WASM
      // inference and parity succeed. Preserve it separately, never hide it.
      const text=m.text(),url=m.location().url;
      if(text.includes('[W:onnxruntime:') && text.includes('Unknown CPU vendor. cpuinfo_vendor value: 0'))runtimeWarnings.push(text);
      else if(url==='https://allwellll.github.io/favicon.ico' && text.includes('status of 404'))resourceWarnings.push({url,status:404,message:text});
      else consoleErrors.push(text);
    });
    page.on('response',r=>{if(r.status()>=400)httpErrors.push({url:r.url(),status:r.status()});});
    const cdp=await context.newCDPSession(page);
    await cdp.send('Network.enable');await cdp.send('Network.setCacheDisabled',{cacheDisabled:true});
    page.on('response',r=>{if(r.url().includes('/models/bun392-'))downloads.push({url:r.url(),status:r.status()});});
    await page.addInitScript(()=>{window.setInterval=cb=>{window.appTick=cb;return 1;};});
    await page.goto(base,{waitUntil:'domcontentloaded'});
    const buildInfo=await page.evaluate(async()=>await(await fetch('build-info.json?verify='+Date.now(),{cache:'no-store'})).json());
    if(process.env.EXPECTED_HEAD)assert.equal(buildInfo.commit,process.env.EXPECTED_HEAD);
    await page.waitForFunction(id=>window.appTick && Array.from(document.querySelectorAll('#model-options option')).some(o=>o.value==='model:'+id),id,{timeout:120000});
    const manifest=await page.evaluate(async()=>await(await fetch('models.json',{cache:'no-store'})).json());
    const row=manifest.models.find(r=>r.id===id);assert.ok(row);
    await page.evaluate(()=>{
      window.modelCalls=[];
      const forward=QQT.TransformerModel.prototype.forward;
      QQT.TransformerModel.prototype.forward=function(...args){
        const result=forward.apply(this,args);
        if(this.meta.embed===392){
          if(![...result.move,...result.bomb,result.value].every(Number.isFinite))throw Error('nonfinite inference');
          modelCalls.push({embed:this.meta.embed,obs:args[0].length,state:args[1].length});
        }
        return result;
      };
      if(window.QQTModelWorker) {
        const load=QQTModelWorker.load;
        QQTModelWorker.load=async(...args)=>{
          const model=await load(...args),forward=model.forward;
          window.loadedModel=model;
          model.forward=async(...inputs)=>{const result=await forward(...inputs);
            if(![...result.move,...result.bomb,result.value].every(Number.isFinite))throw Error('nonfinite inference');
            modelCalls.push({embed:model.meta.embed,obs:inputs[0].length,state:inputs[1].length});return result;};
          return model;
        };
      }
      const reset=QQT.Sim.prototype.reset;
      QQT.Sim.prototype.reset=function(...args){window.appSim=this;return reset.apply(this,args);};
    });
    await page.click('#advanced-open');
    await page.fill('#settings-password', 'demaxiya');
    await page.click('#password-form button[type="submit"]');
    await page.selectOption('#opponent','model:'+id);
    await page.waitForFunction(name=>{
      const status=document.querySelector('#model-status').textContent;
      return status.startsWith('加载失败：') || (status.startsWith('已加载：'+name) && !document.querySelector('#opponent').disabled);
    },row.display_name,{timeout:loadTimeout});
    assert.ok((await page.textContent('#model-status')).startsWith('已加载：'+row.display_name),await page.textContent('#model-status'));
    await page.evaluate(async()=>{for(let i=0;i<16;i++)await appTick();});
    const evidence=await page.evaluate(()=>({calls:modelCalls,modelStatus:document.querySelector('#model-status').textContent,
      strategy:document.querySelector('#opponent').value,team:document.querySelector('#team-mode').value,
      tick:appSim.t,nativeBun:appSim.isBun,nativeItems:appSim.nativeItems,nativeTrap:appSim.nativeTrap,players:appSim.nPlayers,
      loadingComplete:document.querySelector('#loading').classList.contains('done'),
      backend:window.loadedModel?.backend,encoding:window.loadedModel?.meta.bun_state_encoding,
      inferenceTimes:window.loadedModel?.inferenceTimes}));
    assert.ok(evidence.calls.length>=8 && evidence.tick>=16 && evidence.nativeBun && evidence.nativeItems && evidence.nativeTrap && evidence.loadingComplete);
    assert.equal(evidence.players,2);assert.equal(evidence.team,'1v1');
    assert.ok(evidence.calls.every(c=>c.obs===24*13*15 && c.state===24));
    if(row.onnx?.meta.bun_state_encoding)assert.equal(evidence.encoding,row.onnx.meta.bun_state_encoding);
    if(row.onnx)assert.equal(evidence.backend,'worker-wasm');
    const hunterSmoke=await page.evaluate(async()=>{
      if(!window.loadedModel)return [];
      const level=(await(await fetch('assets/maps/levels.json')).json()).find(l=>l.qqt_id===806);
      const rows=[];
      for(const [i,difficulty] of ['hard','normal','easy'].entries()) {
        const sim=new QQT.Sim(153600+i);sim.reset(level,{nativeTrap:true,nativeItems:true});sim.maxSteps=300;
        const bot=new QQTBunHunterBot.BunHunterBot({difficulty,seed:153600+i});const rng=QQT.mulberry32(153600+i);
        let calls=0;
        const step=async settling=>{
          const own=await loadedModel.act(sim,0,rng),enemy=bot.act(sim,1);
          if(!own.every(Number.isFinite))throw Error('Nonfinite Hunter smoke action');
          sim.step(settling?[[own[0],0,0],[enemy[0],0,0]]:[own,enemy]);calls++;
        };
        while(!sim.done&&sim.t<300)await step(false);
        const primary=sim.t,hasHazards=()=>sim.fuse.some(v=>v>0)||sim.blastLinger.some(v=>v>0)||sim.trapped.some(v=>v>0);
        sim.maxSteps=primary+125;
        while(hasHazards()&&sim.t<primary+124){sim.done=false;await step(true);}
        if(hasHazards())throw Error('Unsettled Hunter smoke danger');
        rows.push({difficulty,primary_ticks:primary,residual_ticks:sim.t-primary,inference_calls:calls,
          native_bun:sim.isBun,native_trap:sim.nativeTrap,native_items:sim.nativeItems,
          hazards_remaining:false,technical_smoke_only:true});
      }
      return rows;
    });
    if(row.onnx)assert.equal(hunterSmoke.length,3);
    assert.ok(downloads.length>0 && downloads.every(r=>r.status===200 || r.status===206));assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
    assert.deepEqual(consoleErrors,[]);assert.deepEqual(httpErrors,[]);
    const result={passed:true,url:base,buildInfo,row,downloads,evidence,hunterSmoke,errors,consoleErrors,runtimeWarnings,resourceWarnings,httpErrors,writes,networkMock:false,cacheDisabled:true,requestInterception:false,http1Only};
    fs.mkdirSync(require('path').dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
    await page.screenshot({path:output+'.png',fullPage:true});console.log(JSON.stringify({passed:true,model:id,ticks:evidence.tick,inferenceCalls:evidence.calls.length}));
  } finally {await browser.close();}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
