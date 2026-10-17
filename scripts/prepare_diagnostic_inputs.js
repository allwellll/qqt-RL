'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const source=process.env.H7_COMPLETE_ROOT;
assert(source,'H7_COMPLETE_ROOT must point to the completed H7 namespace');
const complete=JSON.parse(fs.readFileSync(path.join(source,'CAMPAIGN_COMPLETE.json')));
assert(complete.completed && complete.actual_generation===32);
const cp=path.join(source,'formal/bun392_bridge_pfsp8/gen000032');
const proof=JSON.parse(fs.readFileSync(path.join(cp,'COMMITTED.json')));
assert(proof.passed && proof.generation===32);
assert.equal(sha(fs.readFileSync(path.join(cp,'policy.pt'))),proof.policy_sha256);
const modelRoot=path.join(source,'authority_je09_v1/bun392_bridge_pfsp8_g000032/model');
const bytes=fs.readFileSync(path.join(modelRoot,'model.json')),doc=JSON.parse(bytes);
assert.equal(doc.meta.checkpoint_sha256,proof.policy_sha256);
const onnx=fs.readFileSync(path.join(modelRoot,'model.onnx'));
assert.equal(sha(onnx),doc.meta.onnx_sha256);
const out='runs/replay_review_20261017/inputs';fs.mkdirSync(out,{recursive:true});
if(process.env.H7_CODE_ROOT){
  for(const folder of ['scripts','jax_bomb'])fs.cpSync(path.join(process.env.H7_CODE_ROOT,folder),path.join(out,'authority',folder),{recursive:true,filter:p=>!p.includes('__pycache__')&&(fs.statSync(p).isDirectory()||/\.(py|js)$/.test(p))});
  const target=path.join(out,'authority/web/assets/maps');fs.mkdirSync(target,{recursive:true});
  fs.copyFileSync('web/assets/maps/levels.json',path.join(target,'levels.json'));
}
fs.copyFileSync(path.join(cp,'policy.pt'),path.join(out,'policy.pt'));
fs.writeFileSync(path.join(out,'model.json'),bytes);fs.writeFileSync(path.join(out,'model.onnx'),onnx);
const seal=JSON.parse(fs.readFileSync(path.join(source,'SOURCE_SEAL.json')));
const sourceHashes={};
for(const [name,expected] of Object.entries(seal.sources)) {
  const file=path.join(source,'inputs/sealed_sources',name);
  if(!fs.existsSync(file))continue;
  const data=fs.readFileSync(file);assert.equal(sha(data),expected,name);
  const target=path.join(out,'authority',name);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,data);sourceHashes[name]=expected;
}
const receipt={schema:'qqt.h7-input-proof/v1',generation:32,checkpoint_sha256:proof.policy_sha256,
  committed_sha256:sha(fs.readFileSync(path.join(cp,'COMMITTED.json'))),model_sha256:sha(bytes),onnx_sha256:sha(onnx),
  complete_sha256:sha(fs.readFileSync(path.join(source,'CAMPAIGN_COMPLETE.json'))),source_hashes:sourceHashes,cpu_only:true};
fs.writeFileSync(path.join(out,'proof.json'),JSON.stringify(receipt,null,2)+'\n');console.log('Verified and copied immutable H7 g32 policy, CPU model and sealed sources');
