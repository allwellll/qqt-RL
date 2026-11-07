'use strict';
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto'), assert = require('node:assert/strict');
const {createServer} = require('../web/server');
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
(async () => {
  const server = createServer();
  if(process.env.SPECIALIST_FIXTURE_DIR){const original=server.listeners('request')[0];server.removeAllListeners('request');server.on('request',(req,res)=>{const m=/^\/models\/integration-fixture\.(json|onnx)(?:\?|$)/.exec(req.url);if(m){res.writeHead(200,{'content-type':m[1]==='json'?'application/json':'application/octet-stream'});return fs.createReadStream(path.join(process.env.SPECIALIST_FIXTURE_DIR,'model.'+m[1])).pipe(res);}original(req,res);});}
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const publicUrl = process.env.SPECIALISTS_PUBLIC_URL, base = publicUrl || `http://127.0.0.1:${server.address().port}/`;
  const fixtureDir = process.env.SPECIALIST_FIXTURE_DIR, tag = publicUrl ? 'public' : 'local';
  const out = process.env.SPECIALIST_BROWSER_OUTPUT || '.hermes'; fs.mkdirSync(out, {recursive: true});
  const opts = {headless: true, executablePath: chromium.executablePath(), args: ['--no-sandbox', '--disable-gpu']};
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  if (proxy) {const u = new URL(proxy); opts.proxy = {server: u.protocol + '//' + u.host, bypass: '127.0.0.1,localhost'};}
  const results = []; let browser;
  try {
    browser = await chromium.launch(opts);
    for (const width of [1280, 390]) {
      const context = await browser.newContext({viewport: {width, height: 1000}, hasTouch: width === 390});
      const page = await context.newPage(), errors = [], requests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', msg => {if(msg.type()==='error')console.log('browser console',msg.text().slice(0,200));});
      page.on('request', request => {if (request.url().includes('/web-models/')) requests.push(request.url());});
      if (!publicUrl) await context.route('http://127.0.0.1:*/**', async route => {
        const response = await fetch(route.request().url());
        await route.fulfill({status: response.status, headers: Object.fromEntries(response.headers), body: Buffer.from(await response.arrayBuffer())});
      });
      await context.route('**/*.supabase.co/**', route => route.fulfill({json: []}));
      if (fixtureDir) {
        const bytes = fs.readFileSync(path.join(fixtureDir, 'model.json')), graph = fs.readFileSync(path.join(fixtureDir, 'model.onnx'));
        const document = JSON.parse(bytes), fixture = {
          id: 'integration-fixture', display_name: '导出对接样例（非候选）', candidate: 'integration-only', cycle: 0,
          file: 'integration-fixture.json', sha256: hash(bytes), bytes: bytes.length,
          source_sha256: document.meta.checkpoint_sha256, rule_version: 'bun-native-v3-items',
          onnx: {file: 'integration-fixture.onnx', sha256: hash(graph), bytes: graph.length, meta: document.meta,
            reference_zero: JSON.parse(fs.readFileSync(path.join(fixtureDir, 'PARITY.json'))).reference_zero},
          inference: {greedy: false, temperature: .3, seed: 5329236},
          specialist: {label: '界面对接测试', strengths: ['仅检查长处说明的展示'], weaknesses: ['不是训练候选，不用于强弱比较'],
            evidence: {ruleset: 'bun-native-v3-items', games: 1, summary: '合成界面测试字段，不是实际评测成绩', url: 'specialists/integration.json#fixture'}},
        };
        await context.route('**/models.json', async route => {
          const response = publicUrl ? await route.fetch() : null;
          const document = response ? await response.json() : JSON.parse(fs.readFileSync('web/models.json'));
          document.models.push(fixture); await route.fulfill({json: document});
        });
        await context.route('**/models/integration-fixture.onnx', route => route.fulfill({contentType: 'application/octet-stream', body: graph}));
        await context.route('**/models/integration-fixture.json', route => route.fulfill({contentType: 'application/json', body: bytes}));
        await context.route('**/specialists/integration.json', route => route.fulfill({json: {purpose: 'integration fixture only'}}));
      }
      await context.route('**/model_loader.js', async route => {
        const response = publicUrl ? await route.fetch() : null;
        const source = response ? await response.text() : fs.readFileSync('web/model_loader.js', 'utf8');
        await route.fulfill({contentType: 'text/javascript', body: source + '\n;globalThis.__sources=[];const originalDownload=QQTModelLoader.downloadModelAsset;QQTModelLoader.downloadModelAsset=(asset,onProgress,options={})=>originalDownload(asset,onProgress,{...options,onSource:e=>globalThis.__sources.push(e)});'});
      });
      await page.addInitScript(() => {
        let worker, visual;
        Object.defineProperty(window, 'QQTModelWorker', {get: () => worker, set: value => {
          worker = value; const load = value.load; value.load = async (...args) => {const result = await load(...args); window.observedModel = result; return result;};
        }});
        Object.defineProperty(window, 'QQTVisual', {get: () => visual, set: value => {
          visual = value; const create = value.createRenderer; value.createRenderer = (...args) => {const renderer = create(...args), render = renderer.render;
            renderer.render = (sim, ...rest) => {window.observedSim = sim; return render(sim, ...rest);}; return renderer;};
        }});
      });
      await page.goto(base, {waitUntil: 'networkidle', timeout: 60000});
      await page.waitForFunction(() => document.querySelector('#loading')?.classList.contains('done'));
      const defaults = await page.evaluate(async () => (await fetch('model_inference_defaults.json')).json());
      assert.equal(defaults.default_model_id, 'bun392-g-frozen'); assert.equal(defaults.policies['bun392-g-frozen'].temperature, .3);
      const rows = await page.evaluate(async () => (await fetch('models.json')).json());
      const specialists = rows.models.filter(row => row.specialist); assert(specialists.length >= 1);
      // The advanced-settings authorization is outside this test's scope; expose
      // its original controls locally, then use real native select/change events.
      await page.locator('#advanced-settings').evaluate(element => {element.hidden = false;});
      for (const row of specialists) {
        await page.locator('#opponent').selectOption('model:' + row.id);
        console.log('selected',row.id);
        await page.waitForFunction(id => window.observedModel && document.querySelector('#model-status').textContent.startsWith('已加载：')
          && document.querySelector('#opponent').value === 'model:' + id, row.id, {timeout: 90000});
        await page.waitForFunction(() => observedModel.inferenceTimes.length >= 3, null, {timeout: 60000});
        const info = await page.evaluate(() => ({text: document.querySelector('#model-specialty').textContent,
          visible: !document.querySelector('#model-specialty').hidden, backend: observedModel.backend,
          parity: observedModel.parityMaxError, calls: observedModel.inferenceTimes.length,
          rule: QQTRulesets.current, tick: observedSim.t, temperature: observedModel.temperature,
          overflow: document.documentElement.scrollWidth > innerWidth}));
        assert(info.text.includes('bun-native-v3-items') && info.text.includes('v4'));
        assert(info.visible && info.text.includes('长处：') && info.text.includes('短板：') && info.text.includes('证据范围：'));
        assert.equal(info.rule, 'bun-native-v4-facing'); assert(info.calls >= 3 && info.parity < .002 && info.tick > 0);
        assert(!info.overflow); assert((await page.locator('#opponent option:checked').textContent()).includes('专项：'));
        await page.locator('#model-specialty').scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(out, `specialists-${tag}-${width}-${row.id}.png`), fullPage: true});
        const rowSources=(await Promise.all(page.workers().map(worker=>worker.evaluate(()=>globalThis.__sources||[])))).flat();
        assert(rowSources.some(e=>e.status==='verified' && e.url===row.onnx.oss_url),'candidate OSS SHA verification');
        results.push({width, id: row.id, verified_sources:rowSources.filter(e=>e.status==='verified'), purpose: fixtureDir ? 'export/UI integration fixture; not candidate evidence' : 'published specialist', ...info});
      }
      await page.locator('#opponent').selectOption('bun.coop_hunter@hard');
      assert(await page.locator('#model-specialty').isHidden());
      await page.locator('#opponent').selectOption('model:bun392-g-frozen');
      await page.waitForFunction(() => observedModel?.meta?.name === 'g-frozen' && document.querySelector('#model-status').textContent.startsWith('已加载：'), null, {timeout: 600000});
      await page.waitForFunction(() => observedModel.inferenceTimes.length >= 3, null, {timeout: 60000});
      assert(await page.locator('#model-specialty').isHidden());
      const retained = await page.evaluate(() => ({temperature: observedModel.temperature, greedy: observedModel.greedy,
        backend: observedModel.backend, parity: observedModel.parityMaxError, calls: observedModel.inferenceTimes.length, rule: QQTRulesets.current}));
      assert.equal(retained.rule, 'bun-native-v4-facing'); assert.equal(retained.temperature, .3); assert.equal(retained.greedy, false);
      assert(requests.some(url => url.includes('/web-models/bun392-g-frozen-')));
      const verified = (await Promise.all(page.workers().map(worker => worker.evaluate(() => globalThis.__sources || [])))).flat().filter(event => event.status === 'verified');
      assert(verified.some(event => event.url.includes('/web-models/bun392-g-frozen-')));

      results.push({width,verified_sources:verified.map(e=>({url:e.url,status:e.status,source:e.source}))});
      assert.deepEqual(errors, []); results.push({width, retained_default: retained, errors});
      await context.close();
    }
    const result = {status: 'PASS', target: tag, mode: fixtureDir ? 'integration-not-candidate' : 'published-candidates', results};
    fs.writeFileSync(path.join(out, `specialists-${tag}-browser.json`), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally {if (browser) await browser.close(); await new Promise(resolve => server.close(resolve));}
})().catch(error => {console.error(error); process.exitCode = 1;});
