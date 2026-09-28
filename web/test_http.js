#!/usr/bin/env node
'use strict';

const assert = require('assert');
const { createServer } = require('./server.js');

(async () => {
  const server = createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  try {
    const health = await fetch(`http://127.0.0.1:${port}/healthz`);
    assert.strictEqual(health.status, 200);
    assert.deepStrictEqual(await health.json(), { status: 'ok', mode: 'bun' });
    const page = await fetch(`http://127.0.0.1:${port}/`);
    const html = await page.text();
    assert.strictEqual(page.status, 200);
    assert(html.includes('抢包子真人测试'));
    assert(html.includes('app.js'));
    assert(html.includes('model-file'));
    assert(html.includes('published-model'));
    assert(html.includes('model-vs-rule'));
    assert(html.includes('model-progress'));
    const app = await fetch(`http://127.0.0.1:${port}/app.js`);
    assert.strictEqual(app.status, 200);
    const appSource = await app.text();
    assert(appSource.includes('BunRuleTacticalBot'));
    assert(appSource.includes('TransformerModel'));
    const manifestResponse = await fetch(`http://127.0.0.1:${port}/models.json`);
    assert.strictEqual(manifestResponse.status, 200);
    const manifest = await manifestResponse.json();
    assert.strictEqual(manifest.schema, 'qqt.web-models/v1');
    assert.strictEqual(manifest.models.length, 3);
    for (const asset of [
      'visual_renderer.js',
      'tactical_arena.js',
      'controls.js',
      'model_catalog.js',
      'model_loader.js',
      'assets/bg/抢包子.png',
      'assets/%E8%A7%92%E8%89%B24%C3%974%E7%B2%BE%E7%81%B5%E5%9B%BE.png',
      'assets/bomb-custom/%E7%BB%8F%E5%85%B8%E9%BB%84%E6%B3%A1%E6%B3%A1.png',
      'assets/flame/flame_C_1.png',
      'assets/flame/flame_R_6.png',
    ]) {
      const response = await fetch(`http://127.0.0.1:${port}/${asset}`);
      assert.strictEqual(response.status, 200, `${asset} must be deployable`);
      const bytes = await response.arrayBuffer();
      assert(bytes.byteLength > 0, `${asset} must not be empty`);
    }
    const malformed = await fetch(`http://127.0.0.1:${port}/%`);
    assert.strictEqual(malformed.status, 400);
    console.log('Bun-only HTTP smoke 通过');
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
