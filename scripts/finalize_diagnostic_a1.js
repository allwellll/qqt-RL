'use strict';
const fs=require('fs'),assert=require('assert'),crypto=require('crypto'),zlib=require('zlib');
const P=require('../web/review_protocol'),G=require('./generate_diagnostic_replays');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
(async()=>{
  const source=process.env.REPLAY_EPISODES,files=fs.readdirSync(source).filter(f=>/^h8-g32-.*\.json$/.test(f));
  const matrix=JSON.parse(fs.readFileSync('reports/replay_a1_matrix_20261017.json'));
  const docs=files.map(f=>JSON.parse(fs.readFileSync(source+'/'+f))).sort((a,b)=>a.meta.seed-b.meta.seed);
  assert.deepEqual(docs.map(d=>d.meta.seed),matrix.scenes.flatMap(s=>s.seeds));
  const independent=JSON.parse(fs.readFileSync('reports/replay_a1_verification_20261017.json'));assert(independent.passed);
  assert.equal(independent.checkpoint_sha256,matrix.checkpoint_sha256);
  assert.deepEqual(independent.evidence.map(e=>e.replay_id).sort(),docs.map(d=>d.meta.id).sort());
  assert(independent.evidence.every(e=>e.original_checkpoint_critic&&e.cloned_rng&&e.independent_counterfactual_ticks>=8&&e.quantization_error_max<=.50001e-6));
  const loaded=await G.load(),level=require('./eval_web_hunter').loadLevel(),rows=[];
  for(const d of docs){
    assert.equal(d.meta.checkpoint_sha256,matrix.checkpoint_sha256);assert.equal(d.meta.advantage.status,'recorded');
    assert.equal(d.meta.advantage.critic_sha256,independent.critic_sha256);
    d.meta.advantage.calculation_hashes=Object.fromEntries(['scripts/diagnostic_advantage.py','scripts/diagnostic_counterfactual.js','scripts/extract_h8_critic.py','scripts/verify_diagnostic_a1.py'].map(file=>[file,sha(fs.readFileSync(file))]));
    assert.equal(d.advantages.ticks.length,d.actions.length);
    const declared=matrix.scenes.find(s=>s.id===d.meta.scene);assert(declared.seeds.includes(d.meta.seed));
    if(d.meta.scene==='carry_home'){
      const text=fs.readFileSync(process.env.REPLAY_INPUT+'/authority/scripts/je05_web_tasks.js','utf8');
      const body=text.slice(text.indexOf('function setup('),text.indexOf('function defaultLedger('));
      const original=new Function('QQT','authority','assert',body+';return setup;')(require('../web/sim'),require('./eval_web_hunter'),assert);
      assert.deepEqual(P.state(original('carry_home',d.meta.seed)),P.state(P.setup(d.meta,level)),'sealed production task initial state');
    }
    const fresh=await G.episode(d.meta.seed,d.meta.variant,loaded,true,'normal',d.meta.scene);
    assert.deepEqual(fresh.actions,d.actions,'fresh policy and real opponent rerun');
    assert.deepEqual(fresh.checks,d.checks,'state hashes and terminal');
    for(const k of ['ticks','winner','score','stored','done'])assert.deepEqual(fresh.summary[k],d.summary[k]);
    d.meta.selection_rule=matrix.selection;d.meta.rules={native_trap:true,native_items:true,tick_hz:10,max_ticks:d.meta.max_ticks};
    d.meta.scene_protocol_sha256=sha(fs.readFileSync('web/review_protocol.js'));
    d.meta.scene_source=d.meta.scene==='carry_home'?'scripts/je05_web_tasks.js':d.meta.scene==='open'?'scripts/eval_web_hunter.js':'web/sim.js';
    d.meta.scene_source_sha256=sha(fs.readFileSync(d.meta.scene==='carry_home'?process.env.REPLAY_INPUT+'/authority/'+d.meta.scene_source:d.meta.scene_source));
    d.meta.scene_sha256=await P.hash({base_map_sha256:d.meta.map_sha256,scene:d.meta.scene,max_ticks:d.meta.max_ticks,protocol_sha256:d.meta.scene_protocol_sha256});
    d.meta.critic_proof='review-evidence/h8-g32-critic.json';
    d.meta.independent_verification='review-evidence/a1-verification.json';
    d.meta.replay_identity=await P.hash(P.identity(d));P.validate(d);await new P.Player(d,level).verify();
    const bytes=Buffer.from(JSON.stringify(d)),file=d.meta.id+'.json';fs.writeFileSync('web/review-replays/'+file,bytes);
    const row={id:d.meta.id,file,seed:d.meta.seed,variant:d.meta.variant,scene:d.meta.scene,scene_name:d.meta.scene_name,scene_rules:d.meta.scene_rules,map:d.meta.map,map_sha256:d.meta.map_sha256,scene_sha256:d.meta.scene_sha256,model:d.meta.model,opponent:d.meta.opponent,checkpoint_sha256:d.meta.checkpoint_sha256,critic_sha256:d.meta.advantage.critic_sha256,summary:d.summary,sha256:sha(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes,{level:9}).length};
    rows.push(row);console.log(row.id,row.bytes,row.gzip_bytes,'real A1 ticks',d.advantages.ticks.length);
  }
  await loaded.session.release();
  // Start with an active episode that exhibits control/window failure; retain all declared episodes.
  rows.sort((a,b)=>Number(b.summary.failures.length>0)-Number(a.summary.failures.length>0)||a.seed-b.seed);
  const old=JSON.parse(fs.readFileSync('web/review-catalog.json'));
  const history=new Map([...(old.history_replays||[]),...old.replays].filter(r=>!rows.some(n=>n.id===r.id)).map(r=>[r.id,r]));
  const scenes=matrix.scenes.map(s=>({id:s.id,name:s.name,source:s.source,base_map:806}));
  const catalog={schema:'qqt.diagnostic-catalog/v1',model:'H8 v34 g32 raw · 真实 Critic A1',scenes,screened_episodes:docs.length,selection:matrix.selection,predeclared_matrix:'review-evidence/matrix.json',raw_bytes:rows.reduce((n,r)=>n+r.bytes,0),gzip_bytes:rows.reduce((n,r)=>n+r.gzip_bytes,0),replays:rows,history_replays:[...history.values()]};
  fs.writeFileSync('web/review-catalog.json',JSON.stringify(catalog,null,2)+'\n');
  fs.mkdirSync('web/review-evidence',{recursive:true});
  fs.copyFileSync('runs/replay_review_20261017/h8/critic/meta.json','web/review-evidence/h8-g32-critic.json');
  fs.copyFileSync('reports/replay_a1_verification_20261017.json','web/review-evidence/a1-verification.json');
  fs.copyFileSync('reports/replay_a1_matrix_20261017.json','web/review-evidence/matrix.json');
  fs.writeFileSync('reports/replay_a1_export_20261017.json',JSON.stringify({schema:'qqt.a1-export/v1',cpu_only:true,checkpoint_sha256:matrix.checkpoint_sha256,rows,actions_reinferred_exact:true,terminal_state_hashes_exact:true,independent_a1_verified:true},null,2)+'\n');
})().catch(e=>{console.error(e);process.exitCode=1;});
