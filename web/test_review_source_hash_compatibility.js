'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const {verifySource}=require('../scripts/review_source_hash_compatibility');
const entry=require('../scripts/review_source_hash_compatibility.json').entries[0];
const bytes=fs.readFileSync(entry.file),context={batchId:entry.batch_id,field:'reward_hashes'};
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
assert.equal(verifySource(entry.file,hash(bytes),context),null);
for(const field of entry.hash_fields)assert.equal(verifySource(entry.file,entry.published_sha256,{...context,field}),entry);
for(const [file,published,ctx,data]of [
  [entry.file,'a'.repeat(64),context,bytes],
  ['another/file.py',entry.published_sha256,context,bytes],
  [entry.file,entry.published_sha256,{...context,batchId:'other-batch'},bytes],
  [entry.file,entry.published_sha256,{...context,field:'other_hashes'},bytes],
  [entry.file,entry.published_sha256,context,Buffer.concat([bytes,Buffer.from('\n# changed reward\n')])],
  [entry.file,entry.published_sha256,context,Buffer.from(bytes.toString().replace('split(env_key, 13)[11]','split(env_key, 13)[10]'))],
])assert.throws(()=>verifySource(file,published,ctx,data),/Unapproved/);
// Even an accidental future allowlist hash update cannot authorize extra changes.
const altered=Buffer.from(bytes.toString().replace('def init_batch(', 'def changed_init_batch('));
const saved=entry.current_sha256;
try{entry.current_sha256=hash(altered);assert.throws(()=>verifySource(entry.file,entry.published_sha256,context,altered),/outside the approved/);}
finally{entry.current_sha256=saved;}
console.log('Source compatibility GREEN: exact pair and audit-only patch; unknown hashes, scope and reward changes rejected');
