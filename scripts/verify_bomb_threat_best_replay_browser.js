'use strict';
const fs=require('fs'),assert=require('assert'),crypto=require('crypto'),{chromium}=require('/tmp/qqt-review-browser/node_modules/playwright');
const B='runs/bomb_threat_v1/replay_best_u800';
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),page=await browser.newPage({viewport:{width:1450,height:1050}}),errors=[],blocked=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 const url=process.env.REPLAY_URL||'http://localhost:8109/'+B+'/20261029-a3-u800-review.html';
 await page.route('**/*',r=>{if(r.request().url().split('?')[0]===url.split('?')[0]&&r.request().isNavigationRequest()&&r.request().method()==='GET')return r.continue();blocked.push({method:r.request().method(),type:r.request().resourceType()});return r.abort();});
 const response=await page.goto(url),downloadedSha=crypto.createHash('sha256').update(await response.body()).digest('hex');assert.equal(downloadedSha,JSON.parse(fs.readFileSync(B+'/PACKAGE.json')).html_sha256);
 await page.waitForFunction(()=>window.reviewAudit?.verified,{timeout:90000});const rows=[];
 const count=JSON.parse(fs.readFileSync(B+'/MANIFEST.json')).games.length;
 assert.equal(await page.locator('#episode option').count(),count);
 for(let i=0;i<count;i++){
  await page.selectOption('#episode',String(i));await page.waitForFunction(i=>reviewAudit?.key===String(i)&&reviewAudit.verified,i,{timeout:60000});
  await page.locator('#play').click();await page.waitForFunction(()=>reviewAudit.tick>=10,{timeout:15000});await page.locator('#play').click();
  const r=await page.evaluate(()=>({key:reviewAudit.key,index:reviewAudit.index,seat:reviewAudit.seat,verified:reviewAudit.verified,matched:reviewAudit.matched,policy:reviewAudit.policy,model:reviewAudit.model,teacher:reviewAudit.teacher,tick:reviewAudit.tick,short_play_fps:reviewAudit.fps,labels:reviewAudit.labels}));
  assert(r.matched&&r.policy==='8153dafb2b15b7e4be17225bc95cc482417ce0afd8455081234f85f2406b16b6'&&!r.teacher);
  const markers=await page.locator('#diagnostic-timeline button').count();
  if(markers){const first=page.locator('#diagnostic-timeline button').first(),start=Number(await first.getAttribute('data-start'));await first.click();assert.equal(await page.evaluate(()=>reviewAudit.tick),start);r.diagnostic_markers=markers;r.marker_seek_verified=true;}
  await page.locator('#seek').evaluate(e=>{e.value=e.max;e.dispatchEvent(new Event('input'))});assert(await page.evaluate(()=>reviewAudit.tick===Number(document.getElementById('seek').max)));
  rows.push(r);
 }
 await page.selectOption('#episode','1');await page.waitForFunction(()=>reviewAudit?.key==='1');await page.locator('#seek').evaluate(e=>{e.value=120;e.dispatchEvent(new Event('input'))});await page.screenshot({path:B+(process.env.REPLAY_URL?'/remote-browser.png':'/browser.png'),fullPage:true});
 assert.equal(await page.locator('#reward').textContent(),'未记录');assert.equal(await page.locator('main aside canvas').count(),1);
 const result={status:errors.length?'FAIL':'PASS',url,html_sha256:downloadedSha,rows,errors,blocked_requests:blocked,all_entries_full_review_verified:true,all_entries_terminal_seek:true,database_writes_blocked:true,external_dependencies:0,scope:'full hash/action/terminal verification each game,>=10ticks playback each entry; short FPS not sustained benchmark'};
 fs.writeFileSync(B+(process.env.REPLAY_URL?'/REMOTE_BROWSER.json':'/BROWSER.json'),JSON.stringify(result,null,2));await browser.close();console.log(JSON.stringify({status:result.status,entries:rows.length,errors,blocked}));if(errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
