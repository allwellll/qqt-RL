'use strict';
// A staged run substitutes only pending JS; model bytes still come from Pages.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const base='https://allwellll.github.io/qqt-RL/',staged=process.env.RELEASE_MIRROR_STAGED==='1';
 const options={headless:true,executablePath:chromium.executablePath(),args:['--no-sandbox','--disable-gpu']};
 const raw=process.env.HTTPS_PROXY||process.env.https_proxy;
 if(raw){const u=new URL(raw);options.proxy={server:u.protocol+'//'+u.host};}
 const browser=await chromium.launch(options);
 try {
  const context=await browser.newContext({viewport:{width:1280,height:900}}),requests=[],responses=[],errors=[],substituted=[];
  context.on('request',r=>requests.push(r.url()));
  context.on('response',r=>{if(r.url().includes('/models/'))responses.push({url:r.url(),status:r.status()});});
  if(staged)for(const file of ['app.js','model_loader.js','model_catalog.js'])await context.route(base+file,route=>{substituted.push(file);return route.fulfill({path:'web/'+file,contentType:'application/javascript'});});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{let visual;Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:v=>{visual=v;const create=v.createRenderer;v.createRenderer=(...args)=>{const r=create(...args),render=r.render;r.render=(s,...rest)=>{window.mirrorSim=s;return render(s,...rest);};return r;};}});});
  await page.goto(base+'?release-mirror-check='+Date.now(),{waitUntil:'domcontentloaded',timeout:60000});
  await page.waitForFunction(()=>window.mirrorSim?.t>0&&document.querySelector('#loading').classList.contains('done'),null,{timeout:600000});
  const state=await page.evaluate(()=>({selection:document.querySelector('#opponent').value,source:document.querySelector('#loading-source').textContent,marks:performance.getEntriesByType('mark').map(e=>e.name)}));
  assert.equal(state.selection,'model:items-l2-sp-u023302-tau03');
  assert(state.marks.includes('qqt:model-verified'));assert(state.marks.includes('qqt:first-frame'));
  assert(state.source.includes('Release 镜像'));assert(!state.source.includes('主线路'));
  assert(!requests.some(u=>/aliyuncs\.com|\/releases\/download\//.test(u)),'healthy mirror must not request OSS or Release fallback');
  assert(responses.some(r=>r.url.startsWith(base+'models/')&&new URL(r.url).pathname.endsWith('.onnx')&&[200,206].includes(r.status)),'actual Pages ONNX response');
  assert.deepEqual(errors,[]);
  if(staged)for(const file of ['app.js','model_loader.js','model_catalog.js'])assert(substituted.includes(file));
  let sha=null;if(process.env.EXPECTED_SHA){sha=await page.evaluate(async()=>(await(await fetch('build-info.json?fresh='+Date.now(),{cache:'no-store'})).json()).commit);assert.equal(sha,process.env.EXPECTED_SHA);}
  fs.mkdirSync('.hermes',{recursive:true});
  const result={status:'PASS',sha,staged,scope:'real Chromium desktop; staged source substitution only when explicitly enabled; not physical iOS',...state,modelResponses:responses,ossRequests:requests.filter(u=>u.includes('aliyuncs.com')),releaseRequests:requests.filter(u=>u.includes('/releases/download/')),errors};
  fs.writeFileSync(`.hermes/release-mirror-${staged?'staged':'public'}.json`,JSON.stringify(result,null,2)+'\n');
  await page.screenshot({path:`.hermes/release-mirror-${staged?'staged':'public'}.png`});console.log(JSON.stringify({...result,modelResponses:responses.length,modelResponseSample:responses[0]}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
