'use strict';
// Optional private input. Published replays bind to the small source receipt.
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const RECEIPT='web/review-evidence/best-d-20261028/source-receipt.json';
const DEFAULT_MODEL='.hermes/best-d-private/model.json';
function loadOptionalModel({receipt=JSON.parse(fs.readFileSync(RECEIPT)),file=process.env.BEST_D_MODEL||DEFAULT_MODEL,log=console.log}={}){
  assert.match(receipt.model_sha256,/^[a-f0-9]{64}$/,'source receipt model hash');
  if(!fs.existsSync(file)){
    log('SKIP: local <EXTERNAL_MODEL> absent; model-file hash and exact export tensor checks not performed. Source-receipt model_sha256 remains checked.');
    return null;
  }
  const bytes=fs.readFileSync(file),hash=crypto.createHash('sha256').update(bytes).digest('hex');
  assert.equal(hash,receipt.model_sha256,'local <EXTERNAL_MODEL> hash differs from source receipt');
  if(receipt.model_bytes!==undefined)assert.equal(bytes.length,receipt.model_bytes,'local model size');
  log('Local <EXTERNAL_MODEL> SHA-256 matches source-receipt model_sha256.');
  return {bytes,document:JSON.parse(bytes)};
}
module.exports={RECEIPT,DEFAULT_MODEL,loadOptionalModel};
