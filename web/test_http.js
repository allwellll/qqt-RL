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
    const app = await fetch(`http://127.0.0.1:${port}/app.js`);
    assert.strictEqual(app.status, 200);
    const appSource = await app.text();
    assert(appSource.includes('BunRuleTacticalBot'));
    assert(appSource.includes('TransformerModel'));
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
