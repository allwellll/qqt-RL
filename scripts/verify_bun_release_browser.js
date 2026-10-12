'use strict';
// Real catalog, real model download and native 1v1 actions; no result submission.
const assert=require('node:assert/strict'),fs=require('fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base=process.env.WEB_URL || 'https://allwellll.github.io/qqt-RL/';
const output=process.env.EVIDENCE_FILE || 'runs/bun-release-browser.json';
const id='bun392-selfplay3072';
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
  const errors=[],writes=[],downloads=[];
  try {
    const context=await browser.newContext({viewport:{width:1440,height:1000}});
    await context.route('**/*',route=>{
      const request=route.request(),pathname=new URL(request.url()).pathname;
      if(['GET','HEAD'].includes(request.method()) ||
          (request.method()==='POST' && /\/rpc\/qqt_(leaderboard|get_profile|my_win_leaderboard)$/.test(pathname))) return route.continue();
      writes.push(pathname);return route.abort();
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    page.on('response',r=>{if(r.url().includes('/models/bun392-'))downloads.push({url:r.url(),status:r.status()});});
    await page.addInitScript(()=>{window.setInterval=cb=>{window.appTick=cb;return 1;};});
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.appTick && document.querySelector('#model-options option[value="model:bun392-selfplay3072"]'),null,{timeout:120000});
    const manifest=await page.evaluate(async()=>await(await fetch('models.json',{cache:'no-store'})).json());
    const row=manifest.models.find(r=>r.id===id);assert.ok(row);assert.match(row.display_name,/Bun392动态自对弈3072/);
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
    await page.waitForFunction(()=>document.querySelector('#model-status').textContent.startsWith('已加载：Bun392动态自对弈3072') && !document.querySelector('#opponent').disabled,null,{timeout:180000});
    await page.evaluate(async()=>{for(let i=0;i<16;i++)await appTick();});
    const evidence=await page.evaluate(()=>({calls:modelCalls,modelStatus:document.querySelector('#model-status').textContent,
      strategy:document.querySelector('#opponent').value,team:document.querySelector('#team-mode').value,
      tick:appSim.t,nativeBun:appSim.isBun,nativeItems:appSim.nativeItems,nativeTrap:appSim.nativeTrap,players:appSim.nPlayers,
      loadingComplete:document.querySelector('#loading').classList.contains('done')}));
    assert.ok(evidence.calls.length>=8 && evidence.tick>=16 && evidence.nativeBun && evidence.nativeItems && evidence.nativeTrap && evidence.loadingComplete);
    assert.equal(evidence.players,2);assert.equal(evidence.team,'1v1');
    assert.ok(evidence.calls.every(c=>c.obs===24*13*15 && c.state===24));
    assert.ok(downloads.length>0 && downloads.every(r=>r.status===200));assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
    const result={passed:true,url:base,row,downloads,evidence,errors,writes,networkMock:false};
    fs.mkdirSync(require('path').dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
    await page.screenshot({path:output+'.png',fullPage:true});console.log(JSON.stringify({passed:true,model:id,ticks:evidence.tick,inferenceCalls:evidence.calls.length}));
  } finally {await browser.close();}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
