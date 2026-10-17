'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.WEB_URL||'http://127.0.0.1:8098/';
const out=process.env.EVIDENCE_DIR||'runs/replay_review_20261017/v36-browser';
const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json'));
const rows=catalog.replays.filter(r=>r.batch_id==='v36-fixed-eight-20261017');
(async()=>{
  assert.equal(rows.length,8);fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']}),results=[];
  const authSource=fs.readFileSync('web/device_auth.js','utf8'),verifier=authSource.match(/const VERIFIER = '([a-f0-9]+)'/)[1];
  try{for(const mobile of [false,true]){
    const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1600,height:1100},isMobile:mobile,hasTouch:mobile});
    // Reuse the existing static device grant; authentication itself is covered
    // by the unchanged auth regression suite. No password enters evidence.
    await context.addInitScript(rule=>localStorage.setItem('qqt.advanced.authorization',JSON.stringify({schema:'qqt.advanced/v1',rule,granted:true})),verifier);
    const page=await context.newPage(),errors=[],badHttp=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)badHttp.push({path:new URL(r.url()).pathname,status:r.status()});});
    for(const row of rows){
      const d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file)),start=Math.min(2,row.summary.ticks);
      await page.goto(base+'review.html#replay='+row.id+'&tick='+start);
      await page.waitForFunction(id=>document.getElementById('integrity').textContent==='hash 已验证'&&document.getElementById('locator').value.startsWith(id+'@'),row.id,{timeout:120000});
      assert.equal(Number(await page.locator('#seek').inputValue()),start);
      const warning=await page.locator('#episode-warning').textContent();assert(warning.includes('offline_recomputed_current_reward'));assert(warning.includes('非 training-time reward'));assert(warning.includes(d.meta.opponent));assert(!warning.includes('固定向上'));
      assert(await page.locator('#route-experiment').isHidden(),'v35 results must not describe v36');
      assert((await page.locator('#advantage-status').textContent()).includes('无法严格复算'));
      assert.equal(await page.locator('#advantage-track .mark').count(),0);assert.equal(await page.locator('#shaped-advantage-track .mark').count(),0);
      for(const tick of [1,Math.max(1,Math.floor(row.summary.ticks/2)),row.summary.ticks]){
        await page.locator('#seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input'));},tick);
        const approx=d.diagnostic_ticks[tick-1].offline_recomputed_current_reward;
        const expected=Math.abs(approx.approx_total_reward[0])<1e-8?'0':approx.approx_total_reward[0].toFixed(3);
        assert.equal(await page.locator('#reward-delta').textContent(),expected);
        assert((await page.locator('#state-details').textContent()).includes('offline_recomputed_current_reward'));
        assert((await page.locator('#route-status').textContent()).includes('近似诊断'));
        assert.equal(await page.locator('#components .component').count(),20);
        assert.equal(await page.locator('.untranslated').count(),0);
      }
      if(row.seed===2026101781){
        await page.locator('#seek').evaluate(e=>{e.value=714;e.dispatchEvent(new Event('input'));});
        assert((await page.locator('#state-details').textContent()).includes('近似诊断异常'));
        assert((await page.locator('#current-events').textContent()).includes('近似势能跨步不连续'));
      }
      await page.locator('#seek').evaluate(e=>{e.value=2;e.dispatchEvent(new Event('input'));});
      await page.reload();await page.waitForFunction(()=>document.getElementById('integrity').textContent==='hash 已验证',null,{timeout:120000});assert.equal(Number(await page.locator('#seek').inputValue()),2);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
      if(row.seed===2026101781||row.seed===2026101784)await page.screenshot({path:out+`/${mobile?'mobile':'desktop'}-${row.seed}.png`,fullPage:true});
      results.push({seed:row.seed,id:row.id,mobile,deep_link:true,hash_verified:true,reward_tick_checks:3,reload_tick:2,strict_a1_null:true});
      console.log('BROWSER',mobile?'mobile':'desktop',row.seed,'GREEN');
    }
    assert.deepEqual(errors,[]);assert.deepEqual(badHttp,[]);await context.close();
  }}finally{await browser.close();}
  fs.writeFileSync(out+'/checks.json',JSON.stringify({passed:true,base,results},null,2)+'\n');
})().catch(e=>{console.error(e);process.exitCode=1;});
