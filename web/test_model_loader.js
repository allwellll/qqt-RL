'use strict';

const assert = require('assert');
const loader = require('./model_loader.js');

(async () => {
  const chunks = [new Uint8Array([1, 2]), new Uint8Array([3, 4, 5])];
  let index = 0;
  const response = {
    ok: true,
    status: 200,
    headers: { get(name) { return name.toLowerCase() === 'content-length' ? '5' : null; } },
    body: { getReader() { return { async read() { return index < chunks.length ? { done: false, value: chunks[index++] } : { done: true }; } }; } },
  };
  const updates = [];
  const data = await loader.readResponseWithProgress(response, (loaded, total) => updates.push([loaded, total]));
  assert.deepStrictEqual(Array.from(new Uint8Array(data)), [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(updates, [[0, 5], [2, 5], [5, 5]]);
  assert.strictEqual(loader.progressText(2, 5), '下载模型：40%（2 B / 5 B）');
  assert.strictEqual(loader.progressText(1048576, 0), '下载模型：1.0 MiB');
  console.log('网页模型流式下载进度契约通过');
})().catch((error) => { console.error(error); process.exit(1); });
