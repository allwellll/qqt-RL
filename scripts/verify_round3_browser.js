'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8091/',out=path.resolve(process.env.EVIDENCE_DIR||'runs/qqt_round3_20261007/browser-local');
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const evidence=[];
 try{for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
  const context=await browser.newContext({viewport,isMobile:viewport.width<600,hasTouch:viewport.width<600});const page=await context.newPage();
  const errors=[],consoleErrors=[],writes=[],profiles=[],reads=[];
  let failResult=true,failProfile=false,failRead=false,upgrade=true,holdProfile=false,releaseProfile,oldRead,deferNextRead=false,profileSaved=false;
  let row={rank:5,nickname:'旧榜单',victory_message:'旧宣言',best_win_duration_ms:9000,player_ip:'31.*.*.6'};
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')consoleErrors.push(m.text());});
  // Every remote write path is intercepted, including the compatibility fallback.
  const submit=async r=>{
   const p=r.request().postDataJSON().p_payload;writes.push(p);
   if(failResult)return r.fulfill({status:422,json:{error:'controlled'}});
   await new Promise(resolve=>setTimeout(resolve,200));row={...row,rank:1,...(!profileSaved?{nickname:p.nickname,victory_message:p.victory_message}:{})};
   return r.fulfill({json:{level:2,points:3,wins:1,games:1,client_match_id:p.client_match_id,match_upgraded:upgrade,
    match_rank:{rank:1,total:5,percentile:80,comparison:'player-best-v1',result:p.result,mode:p.mode,map_id:p.map_id,difficulty:p.difficulty,opponent:p.opponent,duration_ms:p.game_duration_ms}}});
  };
  await page.route('**/functions/v1/submit-result',submit);await page.route('**/rest/v1/rpc/qqt_submit_result',submit);
  await page.route('**/rest/v1/rpc/qqt_update_profile',async r=>{
   const p=r.request().postDataJSON();profiles.push(p);
   if(failProfile)return r.fulfill({status:422,json:{error:'controlled'}});
   if(holdProfile)await new Promise(resolve=>{releaseProfile=resolve;});
   row={...row,nickname:p.p_nickname,victory_message:p.p_victory_message||row.victory_message};profileSaved=true;
   return r.fulfill({json:{saved:true,nickname:row.nickname,victory_message:row.victory_message}});
  });
  await page.route('**/rest/v1/rpc/qqt_leaderboard',async r=>{
   reads.push({...row});
   // A pre-submit refresh is deliberately stale and late; newer reads finish first.
   if(deferNextRead){deferNextRead=false;oldRead=r;return;}
   await r.fulfill({status:failRead?503:200,json:failRead?{}:[row]});
  });
  await page.addInitScript(()=>{
   window.clockOffset=0;const now=Date.now;Date.now=()=>now()+clockOffset;
   window.setInterval=cb=>{window.appTick=cb;return 1;};window.requestAnimationFrame=cb=>{window.appFrame=cb;return 1;};
  });
  const load=async()=>{
   await page.waitForFunction(()=>window.appFrame,null,{polling:50,timeout:60000});
   await page.evaluate(()=>{const frame=QQT.Sim.prototype.frameStep;QQT.Sim.prototype.frameStep=function(...args){window.appSim=this;return frame.apply(this,args);};appFrame(performance.now());});
   await page.waitForFunction(()=>getComputedStyle(document.getElementById('loading')).opacity==='0',null,{polling:50});
  };
  const action=id=>viewport.width<600?page.locator(id).tap():page.locator(id).click();
  const longPress=async id=>{
   await page.locator(id).scrollIntoViewIfNeeded();const box=await page.locator(id).boundingBox(),x=box.x+box.width/2,y=box.y+box.height/2;
   if(viewport.width<600){const cdp=await context.newCDPSession(page);await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});await page.waitForTimeout(350);await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await cdp.detach();}
   else{await page.mouse.move(x,y);await page.mouse.down();await page.waitForTimeout(350);await page.mouse.up();}
  };
  const render=()=>page.evaluate(()=>appFrame(performance.now()));
  const wait=fn=>page.waitForFunction(fn,null,{polling:50,timeout:20000});
  const finish=async()=>page.evaluate(async()=>{appSim.t=119;appSim.maxSteps=120;clockOffset+=12000;await appTick();appFrame(performance.now());});
  const screenshot=name=>page.screenshot({path:path.join(out,`${viewport.width}-${name}.png`),fullPage:true});
  await page.goto(base);await load();await finish();
  assert.equal(await page.locator('#player-nickname').inputValue(),'');assert(await page.locator('#player-nickname').isVisible());
  assert.equal(await page.locator('#settlement button[type=submit]').count(),1);assert.equal(await page.locator('#settlement-submit').textContent(),'提交');
  assert.equal(await page.locator('#profile-submit,#profile-skip,#leaderboard-profile').count(),0);
  await action('#settlement-submit');assert.equal(writes.length,0,'required nickname prevents writes');
  await page.locator('#player-nickname').fill('初次玩家');await page.locator('#player-message').fill('胜利宣言');
  await screenshot('first-orange-card');
  await action('#settlement-close');await render();assert.equal(writes.length,0);assert.equal(await page.locator('#settlement').isVisible(),false);
  await action('#game');await render();assert.equal(await page.locator('#player-nickname').inputValue(),'初次玩家');
  // R while typing is ignored. R outside input stores draft safely and starts a new game.
  await page.locator('#player-message').focus();await page.keyboard.press('r');await render();assert(await page.evaluate(()=>appSim.done));await page.locator('#player-message').fill('胜利宣言');
  await page.locator('#settlement-close').focus();await page.keyboard.press('r');await render();await wait(()=>{appFrame(performance.now());return !appSim.done;});
  await page.reload();await load();assert.equal(await page.locator('#player-nickname').inputValue(),'初次玩家');assert.equal(await page.locator('#player-message').inputValue(),'胜利宣言');
  await action('#settlement-submit');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent.includes('提交失败');});assert.equal(writes.length,1);
  deferNextRead=true;await action('#leaderboard-retry');
  failResult=false;holdProfile=true;
  await longPress('#settlement-submit');await render();
  await page.locator('#settlement-submit').evaluate(e=>{e.click();e.click();});await page.keyboard.down('Enter');await page.keyboard.down('Enter');await page.keyboard.up('Enter');
  await wait(()=>{appFrame(performance.now());return document.querySelector('#leaderboard-list .rank')?.textContent==='1';});
  assert.equal(writes.length,2);assert.deepEqual(writes[0],writes[1]);
  assert(await page.locator('#settlement-close').isDisabled());
  await page.locator('#player-message').fill('提交中新的宣言');
  await page.evaluate(()=>{window.before=appSim;document.activeElement.blur();window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyR'}));window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyR'}));appFrame(performance.now());});assert(await page.evaluate(()=>appSim===before));
  assert(releaseProfile);holdProfile=false;releaseProfile();
  await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent.includes('新的修改');});
  if(oldRead){await oldRead.fulfill({json:[{rank:5,nickname:'过期榜单',victory_message:'过期宣言'}]});oldRead=null;}
  await page.waitForTimeout(100);assert.equal(await page.locator('#leaderboard-list .rank').textContent(),'1');
  await action('#settlement-submit');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent==='提交成功！';});
  assert.equal(writes.length,2);assert.equal(await page.locator('#leaderboard-list .message').textContent(),'提交中新的宣言');
  await screenshot('success-refreshed');
  // Returning-player card, profile failure, receipt reload and read failure recovery.
  await page.locator('#settlement-close').focus();await page.keyboard.press('r');await render();await finish();
  assert.equal(await page.locator('#player-nickname').inputValue(),'初次玩家');assert.equal(await page.locator('#player-message').inputValue(),'');
  await screenshot('returning-card');failProfile=true;
  await action('#settlement-submit');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent.includes('提交失败');});
  const count=writes.length;failProfile=false;await page.reload();await load();failRead=true;
  await action('#settlement-submit');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent.includes('排行榜刷新失败');});
  assert.equal(writes.length,count,'profile retry never rewrites result');
  assert.equal(await page.locator('#leaderboard-list .message').textContent(),'提交中新的宣言','empty declaration preserves existing server value');
  failRead=false;await action('#leaderboard-retry');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent==='提交成功！';});assert.equal(writes.length,count);
  const card=await page.locator('#settlement').boundingBox(),refresh=await page.locator('#leaderboard-retry').boundingBox();
  assert(card.x+card.width<=refresh.x || card.y+card.height<=refresh.y,'card cannot cover leaderboard refresh');
  // Existing profiles may change only through an upgrade: keep edits for the next one.
  upgrade=false;await page.locator('#settlement-close').focus();await page.keyboard.press('r');await render();await finish();
  await page.locator('#player-nickname').fill('升级后昵称');await page.locator('#player-message').fill('升级后宣言');const beforeProfile=profiles.length;
  await action('#settlement-submit');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent.includes('新资料将在升级时更新');});
  assert.equal(profiles.length,beforeProfile);assert.equal(await page.locator('#leaderboard-list .player').textContent(),'初次玩家');
  upgrade=true;await page.locator('#settlement-close').focus();await page.keyboard.press('r');await render();await finish();
  assert.equal(await page.locator('#player-nickname').inputValue(),'升级后昵称');assert.equal(await page.locator('#player-message').inputValue(),'升级后宣言');
  await action('#settlement-submit');await wait(()=>{appFrame(performance.now());return document.getElementById('settlement-status').textContent==='提交成功！';});
  assert.equal(await page.locator('#leaderboard-list .player').textContent(),'升级后昵称');assert.equal(await page.locator('#leaderboard-list .message').textContent(),'升级后宣言');
  await action('#announcement-open');assert.equal(await page.locator('#announcement-dialog').evaluate(e=>e.open),true);
  await screenshot('announcement');await page.keyboard.press('Escape');assert.equal(await page.evaluate(()=>document.activeElement.id),'announcement-open');
  const bounds=await page.locator('#announcement-open').boundingBox(),aside=await page.locator('aside').boundingBox(),table=await page.locator('.leaderboard').boundingBox();
  assert(bounds.x>aside.x+aside.width/2 && bounds.y+ bounds.height<=table.y,'announcement occupies its own top-right row');
  await action('#advanced-open');await page.locator('#settings-password').fill('wrong');await page.keyboard.press('Enter');assert.match(await page.locator('#password-status').textContent(),/口令不对/);await page.keyboard.press('Escape');
  const botProgress=await page.evaluate(()=>{
   const s=new QQT.Sim(20261007);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','crate','fuse','blastLinger'])s[k].fill(0);
   s.pos.set([10.5,1.5,5.5,8.9]);s.invuln.fill(0);s.spawnProtection.fill(0);s.isBun=true;s.bunInitial=[Infinity,Infinity];
   s.fieldItem[5*QQT.W+9]=2;s.fieldOwner[5*QQT.W+9]=1;s.fieldArmed[5*QQT.W+9]=1;s.crate[5*QQT.W+11]=1;s.crateType[5*QQT.W+11]=0;
   for(let c=5;c<14;c++){s.wall[4*QQT.W+c]=1;s.wall[6*QQT.W+c]=1;}
   const b=new QQTBunCoopHunterBot.BunCoopHunterBot({seed:20261007,difficulty:'hard'}),trace=[];
   for(let t=0;t<16;t++){const a=b.act(s,1);trace.push({tick:s.t,pos:Array.from(s.pos.slice(2,4)),action:a.slice(0,3)});s.step([[4,0,0,1],a]);}
   return {trace,field:s.fieldItem[5*QQT.W+9],alive:s.alive[1],trapped:s.trapped[1]};
  });assert.equal(botProgress.field,0);assert(botProgress.trace.some(r=>r.pos[1]>10));assert(botProgress.alive&&!botProgress.trapped);
  assert.equal(await page.locator('body').textContent().then(t=>/战绩已暂存|队列|局待提交/.test(t)),false);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
  assert(consoleErrors.every(e=>/422|503/.test(e)),JSON.stringify(consoleErrors));
  await screenshot('final');evidence.push({viewport,writesMocked:writes.length,profilesMocked:profiles.length,reads:reads.length,requiredFirstNickname:true,singleAction:true,persistedInputs:true,staleReadRejected:true,refreshFailureRecovered:true,botProgress,errors,consoleErrors});await context.close();
 }}finally{await browser.close();}
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({url:base,kind:'real Chromium/Sim/Canvas/assets; deterministic mocked RPC reads/writes for race/failure; no remote writes',evidence},null,2)+'\n');console.log('Round3 desktop/mobile combined orange card, announcement, leaderboard races/recovery and bot glue progress passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
