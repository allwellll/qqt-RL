'use strict';

(function modelLoaderFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTModelLoader = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildLoader() {
  function sizeText(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1048576).toFixed(1)} MiB`;
  }

  function progressText(loaded, total) {
    if (total > 0) return `下载模型：${Math.floor(loaded * 100 / total)}%（${sizeText(loaded)} / ${sizeText(total)}）`;
    return `下载模型：${sizeText(loaded)}`;
  }

  async function readResponseWithProgress(response, onProgress) {
    if (!response.ok) throw new Error(`模型下载失败 HTTP ${response.status}`);
    const total = Number(response.headers.get('content-length')) || 0;
    onProgress(0, total);
    if (!response.body || !response.body.getReader) {
      const buffer = await response.arrayBuffer();
      onProgress(buffer.byteLength, total || buffer.byteLength);
      return buffer;
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress(loaded, total);
    }
    const output = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.byteLength; }
    return output.buffer;
  }

  async function verifyBuffer(buffer, asset, cryptoApi = globalThis.crypto) {
    if (buffer.byteLength !== asset.bytes) throw new Error('模型大小校验失败');
    const digest = Array.from(new Uint8Array(await cryptoApi.subtle.digest('SHA-256', buffer)))
      .map(value => value.toString(16).padStart(2, '0')).join('');
    if (digest !== asset.sha256) throw new Error('模型 SHA-256 校验失败');
    return buffer;
  }
  return { progressText, readResponseWithProgress, verifyBuffer };
});
