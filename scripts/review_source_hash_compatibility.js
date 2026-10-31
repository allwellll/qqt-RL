'use strict';
// Published identities remain immutable. Only this explicit source pair is compatible.
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const registry=require('./review_source_hash_compatibility.json');
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');

function verifySource(file,publishedHash,{batchId,field},bytes=fs.readFileSync(file)){
  const actual=sha(bytes);
  if(actual===publishedHash)return null;
  const entry=registry.entries.find(e=>e.file===file&&e.published_sha256===publishedHash&&
    e.current_sha256===actual&&e.batch_id===batchId&&e.hash_fields.includes(field));
  assert(entry,`Unapproved ${field} source mismatch: ${file} published=${publishedHash} current=${actual}`);
  const source=bytes.toString('utf8');
  let historical=source;
  for(const patch of entry.patches||[entry]){
    assert.equal(historical.split(patch.new_function).length,2,'Expected exactly one approved audit function');
    historical=historical.replace(patch.new_function,patch.old_function);
  }
  assert.equal(sha(historical),publishedHash,
    'Changes outside the approved audit function are not compatible');
  return entry;
}
module.exports={verifySource};
