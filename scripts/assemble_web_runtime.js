'use strict';
// Same-origin deployment, pinned npm package + lockfile integrity. No CDN or
// cross-origin isolation required: worker uses single-thread SIMD WASM.
const fs=require('fs'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'..');
const source=path.join(root,'node_modules/onnxruntime-web');
assert.equal(JSON.parse(fs.readFileSync(path.join(source,'package.json'))).version,'1.23.2');
const target=path.resolve(process.argv[2]||path.join(root,'web/vendor/ort'));
fs.mkdirSync(target,{recursive:true});
for(const file of ['ort.wasm.min.js','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm']) {
  fs.copyFileSync(path.join(source,'dist',file),path.join(target,file));
}
for (const name of ['onnxruntime_LICENSE.txt','onnxruntime_ThirdPartyNotices.txt']) {
  fs.copyFileSync(path.join(root,'web/third_party',name),path.join(target,name));
}
console.log('Assembled pinned ONNX Runtime Web 1.23.2 (single-thread WASM)');
