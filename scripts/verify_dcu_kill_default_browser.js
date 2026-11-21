'use strict';
// Real published weights and worker inference. No training or DCU use.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||path.resolve('.tools/replay-preview/node_modules/playwright'));
const {createServer}=require('../web/server');
(async()=>{
 const out=path.resolve(process.argv[2]||'runs/dcu-web-u000840-20261115-v3'),publicURL=process.env.DCU_DEFAULT_PUBLIC_URL;
 fs.mkdirSync(out,{recursive:true});
 const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base=publicURL||`http://127.0.0.1:${server.address().port}/`;
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-gpu']});
 const id=process.env.DCU_EXPECTED_MODEL_ID||require('../web/model_inference_defaults.json').mode_defaults.kill;
 const catalog=require('../web/models.json').models;
 const name=catalog.find(row=>row.id===id)?.onnx?.meta?.name;assert(name,'expected model metadata');
 const alternateId=process.env.DCU_ALTERNATE_MODEL_ID;
 const alternateName=catalog.find(row=>row.id===alternateId)?.onnx?.meta?.name;
 if(alternateId)assert(alternateName&&alternateId!==id,'expected alternate model metadata');
 const results=[],responses=new Map();
 try{
  for(const width of [1280,390]){
   const context=await browser.newContext({viewport:{width,height:1000},hasTouch:width===390});
   const page=await context.newPage(),errors=[],requests=[];
   page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
   if(publicURL && process.env.REPLAY_HTTP_BRIDGE==='1'){
    const cache=path.join(out,'http-cache');fs.mkdirSync(cache,{recursive:true});
    await context.route('https://allwellll.github.io/qqt-RL/**',async route=>{
     const url=route.request().url(),key=url+'|'+(route.request().headers().range||'');
     if(responses.has(key)){const result=await responses.get(key);return route.fulfill({status:result.status,headers:result.headers,body:fs.readFileSync(result.file)});}
     const headers=Object.fromEntries(Object.entries(route.request().headers()).filter(([k])=>k==='range'));
     const pending=new Promise((resolve,reject)=>execFile('python',[path.resolve('scripts/dcu_public_http_bridge.py'),url,cache,JSON.stringify(headers)],{maxBuffer:100000,timeout:300000},(e,stdout)=>e?reject(e):resolve(JSON.parse(stdout))));
     responses.set(key,pending);
     try{const result=await pending;await route.fulfill({status:result.status,headers:result.headers,body:fs.readFileSync(result.file)});}
     catch(error){responses.delete(key);console.log('HTTP transport retry:',new URL(url).pathname);await route.abort('failed');}
    });
   }
   // Leaderboard requests are independent of model inference.
   await context.route('**/*.supabase.co/**',route=>route.fulfill({json:[]}));
   await page.addInitScript(({testAdvancedMenu})=>{
    if(testAdvancedMenu){
     let panels;
     Object.defineProperty(window,'QQTPanels',{get:()=>panels,set:value=>{
      panels={...value,mount:doc=>value.mount(doc,{auth:{authorized:()=>true}})};
     }});
    }
    let worker,visual;
    Object.defineProperty(window,'QQTModelWorker',{get:()=>worker,set:value=>{
     worker=value;const load=value.load;value.load=async(...args)=>{const result=await load(...args);window.observedModel=result;return result;};
    }});
    Object.defineProperty(window,'QQTVisual',{get:()=>visual,set:value=>{
     visual=value;const create=value.createRenderer;value.createRenderer=(...args)=>{
      const renderer=create(...args),render=renderer.render;renderer.render=(sim,...rest)=>{window.observedSim=sim;return render(sim,...rest);};return renderer;
     };
    }});
   },{testAdvancedMenu:Boolean(alternateId)});
   const ready=async expected=>{
    await page.waitForFunction(name=>window.observedModel?.meta?.name===name && !document.querySelector('#opponent').disabled && document.querySelector('#loading').classList.contains('done'),expected,{timeout:600000});
    await page.waitForFunction(()=>observedModel.inferenceTimes.length>=3 && observedSim.t>0,null,{timeout:60000});
    return page.evaluate(()=>({model:observedModel.meta.name,backend:observedModel.backend,parity:observedModel.parityMaxError,calls:observedModel.inferenceTimes.length,temperature:observedModel.temperature,greedy:observedModel.greedy,kill:observedSim.killMode,tick:observedSim.t,selected:document.querySelector('#opponent').value,mode:document.querySelector('#game-mode').value,overflow:document.documentElement.scrollWidth>innerWidth}));
   };
   await page.goto(base,{waitUntil:'domcontentloaded',timeout:120000});
   const first=await ready(name);
   console.log('Loaded kill default',width,JSON.stringify(first));
   assert.equal(first.selected,'model:'+id);assert.equal(first.mode,'kill');assert.equal(first.kill,true);
   assert.equal(first.temperature,.3);assert.equal(first.greedy,false);assert.equal(first.backend,'worker-wasm');assert(first.parity<.001);assert(!first.overflow);
   await page.screenshot({path:path.join(out,`default-${publicURL?'public':'local'}-${width}.png`),fullPage:true});
   let alternate=null,switchBack=null;
   if(alternateId){
    await page.locator('#advanced-open').click();
    const selector=page.locator('#advanced-settings #opponent');
    await selector.selectOption('model:'+alternateId);
    alternate=await ready(alternateName);
    assert.equal(alternate.selected,'model:'+alternateId);assert.equal(alternate.kill,true);
    assert.equal(alternate.backend,'worker-wasm');assert.equal(alternate.temperature,.3);
    assert.equal(alternate.greedy,false);assert(alternate.parity<.001);assert(!alternate.overflow);
    await selector.selectOption('model:'+id);
    switchBack=await ready(name);
    assert.equal(switchBack.selected,'model:'+id);assert.equal(switchBack.kill,true);
    assert.equal(switchBack.backend,'worker-wasm');
   }
   await page.locator('#game-mode').selectOption('carry');
   const carry=await ready('L2_SP_u023302');
   assert.equal(carry.selected,'model:items-l2-sp-u023302-tau03');assert.equal(carry.kill,false);assert.equal(carry.backend,'worker-wasm');
   await page.locator('#game-mode').selectOption('kill');await ready(name);
   await page.reload({waitUntil:'domcontentloaded',timeout:120000});const reload=await ready(name);
   assert.equal(reload.selected,'model:'+id);assert.equal(reload.mode,'kill');
   assert(!requests.some(u=>u.includes('aliyuncs.com')));assert.deepEqual(errors,[]);
   results.push({width,first,alternate,switchBack,carry,reload,errors,model_requests:requests.filter(u=>u.includes('/models/'))});await context.close();
  }
  const receipt={status:'PASS_DESKTOP_MOBILE_DEFAULT',url:base,transport:publicURL&&process.env.REPLAY_HTTP_BRIDGE==='1'?'python-stdlib HTTP bridge':'Chromium default',results};
  fs.writeFileSync(path.join(out,publicURL?'PUBLIC_BROWSER.json':'LOCAL_BROWSER.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
