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
  async function downloadVerifiedAsset(url, asset, onProgress=()=>{}, options={}) {
    const fetchAsset=options.fetch || globalThis.fetch;
    const chunkBytes=options.chunkBytes || 262144, concurrency=options.concurrency || 8;
    const pause=options.pause || (ms=>new Promise(resolve=>setTimeout(resolve,ms)));
    if (!Number.isSafeInteger(asset.bytes) || asset.bytes<1 ||
        !Number.isInteger(chunkBytes) || chunkBytes<1 || chunkBytes>1048576 ||
        !Number.isInteger(concurrency) || concurrency<1 || concurrency>8) throw Error('Invalid download shape');
    const controllers=new Set();let failed=false,loaded=0;
    async function range(start,end) {
      for(let attempt=0;attempt<3;attempt++) {
        const controller=new AbortController();controllers.add(controller);
        const timer=setTimeout(()=>controller.abort(),120000);
        let transient=false,delay=1000*2**attempt;
        try {
          const response=await fetchAsset(`${url}${url.includes('?')?'&':'?'}range=${start}`,
            {cache:'no-store',headers:{Range:`bytes=${start}-${end}`},signal:controller.signal});
          if(start===0 && response.status===200) {
            clearTimeout(timer);
            return {full:true,buffer:await readResponseWithProgress(response,onProgress)};
          }
          if(response.status!==206) {
            transient=[429,500,502,503,504].includes(response.status);
            const retryAfter=response.headers.get('retry-after');
            if(retryAfter) {
              const seconds=Number(retryAfter);
              delay=Math.max(delay,Number.isFinite(seconds)?seconds*1000:Date.parse(retryAfter)-Date.now());
              if(!Number.isFinite(delay) || delay>60000)transient=false;
            }
            await response.body?.cancel();
            throw Error(`Model download HTTP ${response.status}`);
          }
          if(response.headers.get('content-range')!==`bytes ${start}-${end}/${asset.bytes}`) throw Error('Model range mismatch');
          const buffer=await response.arrayBuffer();
          if(buffer.byteLength!==end-start+1) throw Error('Model range size mismatch');
          return {buffer};
        } catch(error) {
          if(failed || attempt===2 || !(transient || error instanceof TypeError || error.name==='AbortError'))throw error;
        } finally {clearTimeout(timer);controllers.delete(controller);}
        await pause(delay);
      }
    }
    onProgress(0,asset.bytes);
    try {
      const first=await range(0,Math.min(chunkBytes,asset.bytes)-1);
      if(first.full)return await verifyBuffer(first.buffer,asset,options.cryptoApi);
      const output=new Uint8Array(asset.bytes);output.set(new Uint8Array(first.buffer));
      loaded=first.buffer.byteLength;onProgress(loaded,asset.bytes);
      let next=loaded;
      async function consume() {
        while(!failed && next<asset.bytes) {
          const start=next;next+=chunkBytes;
          const {buffer}=await range(start,Math.min(start+chunkBytes,asset.bytes)-1);
          output.set(new Uint8Array(buffer),start);loaded+=buffer.byteLength;onProgress(loaded,asset.bytes);
        }
      }
      await Promise.all(Array.from({length:concurrency},()=>consume()));
      return await verifyBuffer(output.buffer,asset,options.cryptoApi);
    } catch(error) {failed=true;for(const controller of controllers)controller.abort();throw error;}
  }
  // Public immutable OSS bytes first, original Release second. Pages retains
  // an exact Release mirror for browsers/networks that block Release CORS.
  // Every source must independently pass the same size and SHA verification.
  async function downloadModelAsset(asset, onProgress=()=>{}, options={}) {
    const sources=[...new Set([asset.oss_url, asset.url, `models/${asset.file}`].filter(Boolean))];
    for (const [index,url] of sources.entries()) {
      try {
        options.onSource?.({url,status:'loading'});
        let buffer;
        if (url===asset.oss_url) {
          const controller=new AbortController();
          const timer=setTimeout(()=>controller.abort(),options.timeoutMs || 120000);
          try {
            const response=await (options.fetch || globalThis.fetch)(url,
              {cache:options.cache || 'default',credentials:'omit',signal:controller.signal});
            buffer=await readResponseWithProgress(response,(n)=>onProgress(n,asset.bytes));
            await verifyBuffer(buffer,asset,options.cryptoApi);
          } finally {clearTimeout(timer);}
        } else buffer=await downloadVerifiedAsset(url,asset,onProgress,options);
        options.onSource?.({url,status:'verified'});
        return buffer;
      } catch(error) {
        options.onSource?.({url,status:'failed',reason:error.message});
        if(index===sources.length-1)throw Error(`所有模型下载来源均失败：${error.message}`);
      }
    }
  }
  return { progressText, readResponseWithProgress, verifyBuffer, downloadVerifiedAsset, downloadModelAsset };
});
