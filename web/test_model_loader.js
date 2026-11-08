'use strict';

const assert = require('assert');
const loader = require('./model_loader.js');

(async () => {
  const chunks = [new Uint8Array([1, 2]), new Uint8Array([3, 4, 5])];
  let index = 0;
  const streamResponse = {
    ok: true,
    status: 200,
    headers: { get(name) { return name.toLowerCase() === 'content-length' ? '5' : null; } },
    body: { getReader() { return { async read() { return index < chunks.length ? { done: false, value: chunks[index++] } : { done: true }; } }; } },
  };
  const updates = [];
  const data = await loader.readResponseWithProgress(streamResponse, (loaded, total) => updates.push([loaded, total]));
  assert.deepStrictEqual(Array.from(new Uint8Array(data)), [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(updates, [[0, 5], [2, 5], [5, 5]]);
  assert.strictEqual(loader.progressText(2, 5), '下载模型：40%（2 B / 5 B）');
  assert.strictEqual(loader.progressText(1048576, 0), '下载模型：1.0 MiB');
  const crypto=require('node:crypto'),bytes=Uint8Array.from([1,2,3,4,5,6,7,8]);
  const asset={bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex')};
  let active=0,peak=0,calls=0;
  function response(status,data,range){return {ok:status<400,status,headers:{get:n=>n==='content-range'?range:null},arrayBuffer:async()=>data.buffer};}
  const fetchRange=async(url,options)=>{
    calls++;if(calls===1)return response(503,new Uint8Array());
    const [start,end]=options.headers.Range.match(/\d+/g).map(Number);
    active++;peak=Math.max(peak,active);await new Promise(r=>setTimeout(r,start===2?15:1));active--;
    return response(206,bytes.slice(start,end+1),`bytes ${start}-${end}/${bytes.length}`);
  };
  const options={fetch:fetchRange,cryptoApi:crypto.webcrypto,chunkBytes:2,concurrency:2,pause:async()=>{}};
  const progress=[];
  const ranged=await loader.downloadVerifiedAsset('/model',asset,(n,t)=>progress.push([n,t]),options);
  assert.deepStrictEqual(Array.from(new Uint8Array(ranged)),Array.from(bytes));
  assert.equal(peak,2);assert.equal(calls,5);assert.deepStrictEqual(progress.at(-1),[8,8]);
  await assert.rejects(loader.downloadVerifiedAsset('/model',{...asset,sha256:'0'.repeat(64)},()=>{},options),/SHA-256/);
  await assert.rejects(loader.downloadVerifiedAsset('/model',asset,()=>{},{...options,fetch:async()=>response(206,bytes.slice(0,2),'bytes 1-2/8')}),/range/i);
  await assert.rejects(loader.downloadVerifiedAsset('/model',asset,()=>{},{...options,fetch:async()=>response(206,bytes.slice(0,1),'bytes 0-1/8')}),/size/i);
  let failures=0;
  await assert.rejects(loader.downloadVerifiedAsset('/model',asset,()=>{},{...options,fetch:async()=>{failures++;return response(500,new Uint8Array());}}),/HTTP 500/);
  assert.equal(failures,3);
  failures=0;
  await assert.rejects(loader.downloadVerifiedAsset('/model',asset,()=>{},{...options,fetch:async()=>{failures++;return response(404,new Uint8Array());}}),/HTTP 404/);
  assert.equal(failures,1);
  const fallback=await loader.downloadVerifiedAsset('/model',asset,()=>{},{...options,fetch:async()=>response(200,bytes)});
  assert.deepStrictEqual(Array.from(new Uint8Array(fallback)),Array.from(bytes));
  console.log('网页模型流式下载进度契约通过');
})().catch((error) => { console.error(error); process.exit(1); });

(async()=>{
 const assert=require('node:assert/strict'),crypto=require('node:crypto'),L=require('./model_loader');
 const bytes=Uint8Array.from([9,8,7,6]);
 const a={file:'test.json',bytes:4,sha256:crypto.createHash('sha256').update(bytes).digest('hex'),oss_url:'https://qqt-model.oss-cn-beijing.aliyuncs.com/web-models/test.json',url:'https://github.com/example/repo/releases/download/v1/test.json'};
 const reply=(data,status=200)=>new Response(data,{status,headers:{'Content-Length':data.length}});
 for(const scenario of ['healthy','network','http','size','hash','mirror','all-fail']){
  const calls=[],events=[],progress=[];
  const fetch=async(url,opts)=>{calls.push(url);
   if(url===a.oss_url){assert.equal(opts.credentials,'omit');assert.equal(opts.headers,undefined);
    if(scenario==='network'||scenario==='mirror'||scenario==='all-fail')throw new TypeError('unavailable');
    if(scenario==='http')return reply(bytes,403);if(scenario==='size')return reply(bytes.slice(1));if(scenario==='hash')return reply(Uint8Array.from([0,0,0,0]));return reply(bytes);}
   if(scenario==='all-fail'||(scenario==='mirror'&&url.startsWith(a.url)))throw new TypeError('CORS blocked');
   return reply(bytes);
  };
  const promise=L.downloadModelAsset(a,(n,t)=>progress.push([n,t]),{fetch,cryptoApi:crypto.webcrypto,pause:async()=>{},onSource:e=>events.push(e)});
  if(scenario==='all-fail'){await assert.rejects(promise,/所有模型下载来源/);assert(!events.some(e=>e.status==='verified'));continue;}
  assert.deepEqual(Array.from(new Uint8Array(await promise)),Array.from(bytes));
  assert.equal(calls[0],a.oss_url);assert.deepEqual(progress.at(-1),[4,4]);
  const success=events.find(e=>e.status==='verified').url;
  if(scenario==='healthy'){assert.equal(calls.length,1);assert.equal(success,a.oss_url);assert.deepEqual(events.map(e=>e.status),['loading','verifying','verified']);assert(events.every(e=>e.index===0));}
  else if(scenario==='mirror'){assert(success.startsWith('models/test.json'));assert(calls.some(u=>u.startsWith(a.url)));}
  else {assert(success.startsWith(a.url));assert(!calls.some(u=>u.startsWith('models/')));}
 }
 console.log('OSS preference and verified fallbacks PASS: network/HTTP/size/SHA rejection, Release ordering, CORS mirror, progress and exhausted sources');
})().catch(e=>{console.error(e);process.exitCode=1;});
