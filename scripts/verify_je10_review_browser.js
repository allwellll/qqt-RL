'use strict';
const fs=require('fs'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8099/',out=process.env.EVIDENCE_DIR||'runs/je10_release/browser';
const catalog=require('../web/review-catalog.json'),rows=catalog.replays.filter(r=>r.batch_id==='je10-best-completed-20261018');
(async()=>{
  fs.mkdirSync(out,{recursive:true});assert.equal(rows.length,4);
  const executablePath=process.env.CHROMIUM_PATH||await require(process.env.CHROMIUM_MODULE||'@sparticuz/chromium').executablePath();
  const browser=await chromium.launch({headless:true,executablePath,args:['--no-sandbox','--disable-dev-shm-usage']});
  const verifier=fs.readFileSync('web/device_auth.js','utf8').match(/const VERIFIER = '([a-f0-9]+)'/)[1],results=[];
  try{for(const mobile of [false,true]){
    const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1600,height:1100},isMobile:mobile,hasTouch:mobile});
    await context.addInitScript(rule=>localStorage.setItem('qqt.advanced.authorization',JSON.stringify({schema:'qqt.advanced/v1',rule,granted:true})),verifier);
    const page=await context.newPage(),errors=[],badHttp=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)badHttp.push({path:new URL(r.url()).pathname,status:r.status()});});
    for(const row of rows){
      const d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
      await page.goto(base+'review.html#replay='+row.id+'&tick=2');
      await page.waitForFunction(id=>document.getElementById('integrity').textContent==='hash 已验证'&&document.getElementById('locator').value.startsWith(id+'@'),row.id,{timeout:120000});
      assert.equal(Number(await page.locator('#seek').inputValue()),2);assert(await page.locator('#je10-advantage').isVisible());assert(await page.locator('#route-experiment').isHidden());
      assert((await page.locator('#episode-warning').textContent()).includes('非历史训练 rollout'));
      for(const tick of [1,2,Math.floor(d.actions.length/2),d.actions.length]){
        await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);
        const values=await page.locator('#je10-advantage-details dl > div').evaluateAll(es=>Object.fromEntries(es.map(e=>[e.querySelector('dt').textContent,e.querySelector('dd').textContent])));
        const a=d.diagnostic_ticks[tick-1].approximate_a1;
        for(const [label,k] of [['raw A1（近似）','raw_a1'],['shaped A1（近似）','shaped_a1'],['ΔA = shaped A1 − raw A1（近似）','delta_a'],['raw A1 相邻 tick 增量','raw_a1_increment'],['shaped A1 相邻 tick 增量','shaped_a1_increment'],['ΔA 相邻 tick 增量','delta_a_increment']])assert.equal(values[label],a[k]===null?'未记录':a[k].toFixed(8));
        const state=await page.locator('#state-details').textContent();assert(state.includes(d.diagnostic_ticks[tick-1].state_after_sha256));assert(state.includes('Phi'));assert(state.includes('奖励分量'));
      }
      await page.locator('#seek').evaluate(e=>{e.value=2;e.dispatchEvent(new Event('input'));});await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:120000});assert.equal(Number(await page.locator('#seek').inputValue()),2);
      const overflow=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,elements:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&e.getBoundingClientRect().width>0).slice(0,8).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right}))}));assert(overflow.scroll<=overflow.width+1,JSON.stringify(overflow));
      await page.screenshot({path:out+`/${mobile?'mobile':'desktop'}-${row.task}.png`,fullPage:true});results.push({mobile,id:row.id,deep_link:true,reload_tick:2,visible_a1_checks:4,hash_verified:true});
      console.log(mobile?'mobile':'desktop',row.task,'GREEN');
    }
    // Exercise the actual selector and same-origin, SHA-verified worker/WASM load.
    await page.goto(base);await page.locator('#advanced-open').click();await page.locator('#advanced-settings').waitFor({state:'visible'});
    await page.waitForFunction(()=>document.querySelector('#model-options option')?.value==='model:bun392-je10-v38-r01-g8');
    const selected=await page.locator('#model-options option').first().getAttribute('value');await page.locator('#opponent').selectOption(selected);
    await page.waitForFunction(()=>document.getElementById('model-status').textContent.includes('已加载：Bun392 JE10 v38 r01 g8'),null,{timeout:180000});
    const worker=await page.evaluate(async()=>{const row=(await(await fetch('models.json')).json()).models[0],m=await QQTModelWorker.load(row);const r=await m.forward(new Float32Array(24*13*15),new Float32Array(24));const result={backend:m.backend,parity:m.parityMaxError,finite:[...r.move,...r.bomb,r.value].every(Number.isFinite),checkpoint:m.meta?.checkpoint_sha256};m.close();return result;});
    assert(worker.finite);assert.equal(worker.backend,'worker-wasm');assert(worker.parity<.001);results.push({mobile,model_selector_loaded:true,worker});
    assert.deepEqual(errors,[]);assert.deepEqual(badHttp,[]);await context.close();
  }}finally{await browser.close();}
  fs.writeFileSync(out+'/checks.json',JSON.stringify({passed:true,base,results},null,2)+'\n');
})().catch(e=>{console.error(e);process.exitCode=1;});
