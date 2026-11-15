'use strict';
// Standalone/page replay validation; browser rasterization stays on CPU.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFile}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||path.resolve('.tools/replay-preview/node_modules/playwright'));
(async()=>{
 const target=process.argv[2]||'file://'+path.resolve('runs/kill-replays-20261115-01/watch.html');
 const out=path.resolve(process.argv[3]||'runs/kill-replays-20261115-01');
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-gpu']});
 const errors=[],requests=[];
 async function transport(page){
  if(process.env.REPLAY_HTTP_BRIDGE!=='1')return;
  // Use the container's ordinary HTTP transport without reading its proxy
  // credentials. Preserve production security headers and the actual bytes.
  await page.route('https://allwellll.github.io/qqt-RL/dcu-replays/**',async route=>{
   const cache=path.join(out,'http-cache');fs.mkdirSync(cache,{recursive:true});
   const raw=await new Promise((resolve,reject)=>execFile('python',[path.resolve('scripts/dcu_public_http_bridge.py'),route.request().url(),cache,'{}'],{maxBuffer:100000,timeout:300000},(error,stdout)=>error?reject(error):resolve(stdout)));
   const response=JSON.parse(raw);await route.fulfill({status:response.status,headers:response.headers,body:fs.readFileSync(response.file)});
  });
 }
 try{
  const page=await browser.newPage({viewport:{width:1280,height:1000}});
  await transport(page);
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.goto(target,{timeout:120000});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('验证通过'),null,{timeout:60000});
  const games=await page.locator('#match option').count();assert(games>0);
  assert.equal(games,await page.evaluate(()=>PAYLOAD.document.replays.length));
  assert.deepEqual(await page.locator('canvas').evaluate(c=>[c.width,c.height]),[900,880]);
  assert(!(await page.locator('main').innerText()).includes('训练前'));
  await page.locator('#next').click();assert((await page.locator('#clock').innerText()).includes('1 / 1800'));
  await page.locator('#event').click();const eventClock=await page.locator('#clock').innerText();
  await page.evaluate(()=>{const s=document.querySelector('#seek');s.value='1800';s.dispatchEvent(new Event('input'));});
  assert((await page.locator('#clock').innerText()).includes('1800 / 1800'));
  await page.locator('#back').click();assert((await page.locator('#clock').innerText()).includes('1799 / 1800'));
  await page.selectOption('#match',String(games-1));await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('验证通过'));
  await page.locator('#play').click();await page.waitForTimeout(400);await page.locator('#play').click();
  assert((await page.locator('#clock').innerText())!=='0.0 秒 · 0 / 1800 tick');
  await page.screenshot({path:path.join(out,'preview-desktop.png'),fullPage:true});
  const phone=await browser.newPage({viewport:{width:390,height:844},isMobile:true,deviceScaleFactor:1});
  await transport(phone);
  phone.on('pageerror',e=>errors.push(e.message));await phone.goto(target,{timeout:120000});await phone.waitForFunction(()=>document.querySelector('#status').textContent.includes('验证通过'),null,{timeout:60000});
  assert(await phone.evaluate(()=>document.querySelector('canvas').getBoundingClientRect().right<=innerWidth));
  assert(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await phone.screenshot({path:path.join(out,'preview-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert(!requests.some(u=>/\.onnx|\.wasm|model\.json/.test(u)));
  const result={status:'PASS_DESKTOP_MOBILE_REPLAY',url:target,games,state_sha256_verified:true,transport:process.env.REPLAY_HTTP_BRIDGE==='1'?'python-stdlib HTTP bridge':'Chromium default',
    stepping:true,seeking_terminal:true,next_combat_event:eventClock,switching_games:true,play_pause:true,mobile_layout:true,requests,errors};
  fs.writeFileSync(path.join(out,'BROWSER_CHECK.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
