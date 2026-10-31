'use strict';
// Read-only browser acceptance. PLAYWRIGHT_MODULE may point to an external install.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {createServer}=require('../web/server');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
 const server=process.env.WEB_URL?null:createServer();
 if(server)await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const base=process.env.WEB_URL||`http://127.0.0.1:${server.address().port}/`;
 const output=process.env.BROWSER_RECEIPT||'runs/main_sync_20261031/BROWSER.json';
 const manifest=JSON.parse(fs.readFileSync('web/models.json')),row=manifest.models[0];
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const errors=[],consoleErrors=[],writes=[];
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  await context.route('**/*',route=>{
   const q=route.request();
   if(['GET','HEAD'].includes(q.method()))return route.continue();
   // Allow only read-only RPC names; all other writes are locally blocked.
   if(q.method()==='POST'&&/\/rpc\/qqt_(leaderboard|get_profile|my_win_leaderboard|my_win_leaderboard_ruleset|ruleset_contract)$/.test(new URL(q.url()).pathname))return route.continue();
   writes.push({method:q.method(),path:new URL(q.url()).pathname});
   return route.fulfill({status:200,contentType:'application/json',body:'{}'});
  });
  await context.addInitScript(()=>{const interval=window.setInterval;window.setInterval=function(fn,delay,...args){if(delay===100&&fn.name==='tick'){window.syncTick=fn;return interval(()=>{},delay)}return interval(fn,delay,...args)}});
  const page=await context.newPage();
  const observe=p=>{p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>{if(m.type()==='error'&&!m.text().includes('favicon.ico'))consoleErrors.push(m.text())})};observe(page);
  await page.goto(base,{waitUntil:'networkidle',timeout:120000});
  const identity=await page.evaluate(async()=>({ruleset:QQTRulesets.current,sim:QQT.RULE_VERSION,manifest:await(await fetch('models.json',{cache:'no-store'})).json(),simBytes:Array.from(new Uint8Array(await(await fetch('sim-bun-native-v2.js')).arrayBuffer()))}));
  assert.equal(identity.ruleset,'bun-native-v2');assert.equal(identity.sim,'bun-native-v2');assert.deepEqual(identity.manifest,manifest);
  const simulatorSha=sha(Buffer.from(identity.simBytes));delete identity.simBytes;
  assert.equal(simulatorSha,sha(fs.readFileSync('web/sim-bun-native-v2.js')));
  await page.evaluate(()=>{
   window.syncActions=[];
   const load=QQTModelWorker.load;QQTModelWorker.load=async(...a)=>{const m=await load(...a);window.syncModel=m;const act=m.act;m.act=async(...args)=>{const action=await act.apply(m,args);if(!action.every(Number.isFinite))throw Error('nonfinite action');syncActions.push({seat:args[1],tick:args[0].t,action});return action};return m};
   const reset=QQT.Sim.prototype.reset;QQT.Sim.prototype.reset=function(...a){window.syncSim=this;return reset.apply(this,a)};
   const el=document.getElementById('opponent');el.value='model:'+document.querySelector('#model-options option[value^="model:"]').value.slice(6);el.dispatchEvent(new Event('change'));
  });
  await page.waitForFunction(()=>window.syncModel&&document.getElementById('model-status').textContent.startsWith('已加载：'),null,{timeout:300000});
  for(const close of await page.locator('dialog[open] [data-dialog-close]').all())await close.click();
  await page.keyboard.down('ArrowRight');await page.waitForTimeout(300);await page.keyboard.up('ArrowRight');
  await page.evaluate(async()=>{for(let i=0;i<2400&&!syncSim.done;i++)await syncTick()});
  const game=await page.evaluate(()=>({ticks:syncSim.t,done:syncSim.done,winner:syncSim.winner,actions:syncActions.length,actionSeats:[...new Set(syncActions.map(a=>a.seat))],backend:syncModel.backend,greedy:syncModel.greedy,policySha:syncModel.meta.checkpoint_sha256,nativeItems:syncSim.nativeItems,nativeTrap:syncSim.nativeTrap}));
  assert(game.done);assert(game.actions>0);assert.deepEqual(game.actionSeats,[1]);assert(game.greedy&&game.nativeItems&&game.nativeTrap);assert.equal(game.policySha,row.source_sha256);
  const review=await context.newPage();observe(review);await review.goto(new URL('review.html',base).href,{waitUntil:'networkidle',timeout:120000});
  await review.waitForFunction(()=>!document.getElementById('play').disabled,null,{timeout:180000});
  // Playback smoke opens the diagnostic panel explicitly; password validation is covered separately.
  await review.evaluate(()=>{document.getElementById('advanced-settings').hidden=false});
  const before=await review.textContent('#exact-time');await review.click('#play');await review.waitForTimeout(500);await review.click('#play');const after=await review.textContent('#exact-time');assert.notEqual(before,after);
  await review.evaluate(()=>{const e=document.getElementById('seek');e.value=e.max;e.dispatchEvent(new Event('input'))});
  const replay={before,after,terminal:await review.textContent('#exact-time'),selected:await review.textContent('#selected-replay'),integrity:await review.textContent('#integrity')};
  let build=null;if(process.env.EXPECTED_HEAD){build=await page.evaluate(async()=>await(await fetch('build-info.json?verify='+Date.now(),{cache:'no-store'})).json());assert.equal(build.commit,process.env.EXPECTED_HEAD)}
  assert.deepEqual(errors,[]);assert.deepEqual(consoleErrors,[]);
  fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify({status:'PASS',build,model:row.id,model_sha256:row.sha256,simulator_sha256:simulatorSha,identity,game,replay,errors,consoleErrors,writes_blocked:writes,database_writes_blocked:true,password_gate_tested:false},null,2)+'\n');
  console.log(JSON.stringify({status:'PASS',model:row.id,game,replay,build}));
 }finally{await browser.close();if(server)await new Promise(r=>server.close(r))}
}
main().catch(e=>{console.error(e);process.exitCode=1});
