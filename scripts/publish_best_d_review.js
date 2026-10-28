'use strict';
// Local asset assembly only. Does not publish, push, or alter any play default.
const fs=require('node:fs'),crypto=require('node:crypto'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const P=require('../web/review_protocol'),E=require('../web/review_evaluation');
const {loadOptionalModel,RECEIPT}=require('./best_d_model_input');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const root='web/review-evidence/best-d-20261028',input=process.env.BEST_D_PRIVATE||'.hermes/best-d-private';
async function main(){
  const matrix=JSON.parse(fs.readFileSync(`${root}/matrix.json`)),actor=JSON.parse(fs.readFileSync(`${root}/actor-parity.json`));
  const receipt=JSON.parse(fs.readFileSync(RECEIPT));loadOptionalModel({receipt});
  assert.equal(actor.model_sha256,receipt.model_sha256);assert.equal(matrix.checkpoint_sha256,receipt.policy_sha256);
  assert((actor.passed||actor.numerical_parity_passed)&&actor.actor_state_exact);
  const model={id:matrix.model_id,label:'D · ITER-PBT-V1 slot05 u720 · 评估录像',
    checkpoint:matrix.checkpoint,checkpoint_sha256:matrix.checkpoint_sha256,model_sha256:receipt.model_sha256,
    state_sha256:actor.state_sha256,actor_evidence:`${root.slice(4)}/actor-parity.json`,
    matrix_file:`${root.slice(4)}/matrix.json`,reward_scope:E.SCHEMA,
    training_reward:'unrecorded',raw_a1:'unrecorded',shaped_a1:'unrecorded',delta_a:'unrecorded'};
  const rows=[];
  for(const item of matrix.episodes){
    const id=`d-slot05-u720-${item.scene}-${item.difficulty}-${item.seed}`,file=`${id}.json`;
    const doc=JSON.parse(fs.readFileSync(`${input}/${file}`));
    assert(doc.meta.review_reward,'reward reconstruction must complete before assembly');
    assert.equal(doc.meta.matrix_sha256,sha(fs.readFileSync(`${root}/matrix.json`)));
    doc.meta.onnx_sha256=actor.wasm_graph_sha256;
    doc.meta.source_hashes=Object.fromEntries(['web/sim.js','web/bun_hunter_bot.js','web/bot_contract.js','web/review_protocol.js','scripts/generate_best_d_review.js','scripts/publish_best_d_review.js','scripts/best_d_model_input.js'].map(f=>[f,sha(fs.readFileSync(f))]));
    doc.meta.replay_identity=await P.hash(P.identity(doc));P.validate(doc);E.validate(doc,model);
    await new P.Player(doc,require('./eval_web_hunter').loadLevel()).verify();
    const bytes=Buffer.from(JSON.stringify(doc));fs.writeFileSync(`web/review-replays/${file}`,bytes);
    rows.push(Object.fromEntries([...['id','seed','scene','scene_name','scene_rules','model','model_id','opponent','checkpoint_sha256','model_sha256','map','map_sha256','batch_id'].map(k=>[k,doc.meta[k]]),
      ['file',file],['summary',doc.summary],['sha256',sha(bytes)],['bytes',bytes.length],['gzip_bytes',zlib.gzipSync(bytes,{level:9}).length]]));
  }
  const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json'));
  catalog.evaluation={schema:'qqt.review-evaluation-catalog/v1',scope:'Reward-only CPU evaluation; independent of complete-Critic diagnostic admission; all seeds retained; existing default unchanged.',models:[model],replays:rows};
  fs.writeFileSync('web/review-catalog.json',JSON.stringify(catalog,null,2)+'\n');
  const groups=[];
  for(const item of matrix.episodes){
    const key=`${item.scene}/${item.difficulty}`;
    if(groups.some(g=>g.key===key))continue;
    const selected=rows.filter(r=>r.scene===item.scene&&r.opponent===`bun.hunter/${item.difficulty}`);
    groups.push({key,seeds:selected.map(r=>r.seed),games:selected.length,
      win:selected.filter(r=>r.summary.outcome==='win').length,loss:selected.filter(r=>r.summary.outcome==='loss').length,draw:selected.filter(r=>r.summary.outcome==='draw').length,
      death:selected.reduce((n,r)=>n+r.summary.death[0],0),delivery:selected.reduce((n,r)=>n+r.summary.delivery[0],0)});
  }
  const verification={schema:'qqt.best-d-review-verification/v1',passed:true,all_seeds_retained:true,episodes:rows.length,
    checkpoint_sha256:model.checkpoint_sha256,model_sha256:model.model_sha256,
    matrix_sha256:sha(fs.readFileSync(`${root}/matrix.json`)),actor_parity_sha256:sha(fs.readFileSync(`${root}/actor-parity.json`)),
    outcome_definition:'native/carry_home: production bun winner; open: any learner death is loss, otherwise exclusive opponent kill is win, otherwise draw; simulator bun winner remains separate.',
    groups,rows:rows.map(r=>({id:r.id,seed:r.seed,scene:r.scene,opponent:r.opponent,sha256:r.sha256,bytes:r.bytes,...r.summary}))};
  fs.writeFileSync(`${root}/verification.json`,JSON.stringify(verification,null,2)+'\n');
  console.log('Local D assets assembled and replay-verified',rows.length,JSON.stringify(groups));
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
