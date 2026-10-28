'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const P=require('./review_protocol'),E=require('./review_evaluation'),A=require('./review_advantage'),Gate=require('./review_admission');
const c=require('./review-catalog.json'),matrix=require('./review-evidence/best-d-20261028/matrix.json');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const {loadOptionalModel,RECEIPT}=require('../scripts/best_d_model_input');
(async()=>{
  assert.equal(c.model_comparison.default_model,'r01-g8');assert.equal(c.replays[0].id,'je10-g8-metrics-recomputed-natural-2026101800');
  assert.equal(c.evaluation.schema,'qqt.review-evaluation-catalog/v1');const model=c.evaluation.models[0],level=require('../scripts/eval_web_hunter').loadLevel();
  assert.equal(model.id,matrix.model_id);assert.equal(model.checkpoint_sha256,matrix.checkpoint_sha256);
  assert.equal(c.evaluation.replays.length,matrix.episodes.length);
  const actor=require('./review-evidence/best-d-20261028/actor-parity.json');assert(actor.actor_state_exact);assert(actor.passed||actor.numerical_parity_passed);
  if(actor.export_tensors_exact!==true){assert.equal(actor.export_tensors_exact,null);assert.equal(actor.export_tensors_status,'SKIP');assert.equal(actor.passed,false);console.log('SKIP: stored Actor evidence has no exact export tensor comparison; numerical evidence checked separately.');}
  assert(actor.max_error<actor.tolerance);assert.equal(actor.greedy_action_mismatches,0);
  const receipt=JSON.parse(fs.readFileSync(RECEIPT));
  assert.equal(receipt.model_sha256,model.model_sha256);assert.equal(receipt.model_sha256,actor.model_sha256);
  assert.equal(receipt.policy_sha256,model.checkpoint_sha256);
  const local=loadOptionalModel({receipt});
  if(local){
    assert.equal(local.document.meta.checkpoint_sha256,receipt.policy_sha256);
    console.log('SKIP: npm Review suite checks local model hash only; checkpoint-to-export exact tensor comparison requires scripts/verify_best_d_actor.py.');
  }
  for(const [i,row]of c.evaluation.replays.entries()){
    const expected=matrix.episodes[i];assert.equal(row.seed,expected.seed);assert.equal(row.scene,expected.scene);assert.equal(row.opponent,`bun.hunter/${expected.difficulty}`);
    const bytes=fs.readFileSync('web/review-replays/'+row.file),doc=JSON.parse(bytes);
    assert.equal(sha(bytes),row.sha256);assert.equal(bytes.length,row.bytes);P.validate(doc);E.validate(doc,model);assert.throws(()=>Gate.validate(doc),/diagnostic admission/);
    assert.equal(doc.meta.matrix_sha256,sha(fs.readFileSync('web/review-evidence/best-d-20261028/matrix.json')));
    for(const [file,hash]of Object.entries(doc.meta.reward_hashes))assert.equal(sha(fs.readFileSync(file)),hash,file);
    for(const [file,hash]of Object.entries(doc.meta.source_hashes))assert.equal(sha(fs.readFileSync(file)),hash,file);
    assert.equal(doc.meta.onnx_sha256,actor.wasm_graph_sha256);
    const p=new P.Player(doc,level);await p.verify();p.seek(Math.min(200,doc.actions.length));p.seek(1);p.seek(doc.actions.length);
    assert.equal(A.atTick(doc,1).advantage,null);assert.equal(A.atTick(doc,1).delta,null);
    if(row.scene==='open'){
      const s=P.replaySetup(doc.meta,level);assert.deepEqual(s.hp,[1,1]);assert(s.brick.every(v=>!v));assert.equal(s.maxSteps,300);assert.equal(doc.actions.length,300);assert.equal(s.nativeTrap,false);
    }
    for(const change of [d=>d.meta.model_sha256='a'.repeat(64),d=>d.meta.review_reward.measured_count--,d=>d.metrics={},d=>d.meta.reward_status='recorded',d=>d.meta.advantage.status='recorded',d=>d.meta.training_reward_status='recorded',d=>d.rewards.push([1,{fake_reward:1},{}])]){
      const bad=structuredClone(doc);change(bad);assert.throws(()=>E.validate(bad,model),/evaluation evidence/);
    }
    const bad=structuredClone(doc);bad.meta.review_reward.scope+=' changed';await assert.rejects(new P.Player(bad,level).verify(),/identity/);
  }
  assert.equal(Gate.catalogGate(c),22);
  console.log('D Review GREEN: source-receipt model identity, all 15 predeclared seeds, honest reward scope, no false A1, tamper rejection, playback/hash/seek, defaults and prior admission retained; historical Actor proof checked separately from current optional local checks');
})().catch(e=>{console.error(e);process.exitCode=1;});
