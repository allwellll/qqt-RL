'use strict';
// Historical compatibility fixtures remain verified independently of public admission.
// The current catalog is exhaustively gated by test_review_admission.js.
const assert=require('assert'),fs=require('fs'),crypto=require('crypto'),zlib=require('zlib');
const D=require('./review_dagger'),P=require('./review_protocol'),E=require('../scripts/export_diagnostic_dagger');
const read=file=>JSON.parse(fs.readFileSync('web/review-dagger/'+file));
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
(async()=>{
  const catalog=read('catalog.json');assert.equal(catalog.schema,'qqt.dagger-catalog/v1');
  for(const row of catalog.runs){
    const bytes=fs.readFileSync('web/review-dagger/'+row.file);assert.equal(bytes.length,row.bytes);assert.equal(sha(bytes),row.sha256);
    assert.equal(zlib.gzipSync(bytes,{level:9}).length,row.gzip_bytes);const doc=D.validateQuality(JSON.parse(bytes));
    for(const source of doc.sources){
      assert.equal(sha(fs.readFileSync('web/review-dagger/sources/'+source.file)),source.sha256);
      if(process.env.SOURCE_DAGGER_ROOT)assert.equal(sha(fs.readFileSync(require('path').join(process.env.SOURCE_DAGGER_ROOT,source.path))),source.original_sha256||source.sha256,'read-only original artifact hash');
    }
    for(const key of [...Object.keys(doc.funnel),...Object.keys(doc.teacher_quality),...Object.keys(doc.refusal_reasons),...Object.keys(doc.smoke_a||{})])assert(D.FIELDS[key],`untranslated key ${key}`);
    assert.equal(doc.funnel.explicit_localization,null);assert.equal(doc.teacher_quality.counterfactual_survival,null);
    assert.equal(doc.refusal_reasons.AMBIGUOUS,null);
  }
  const h6=read('h6-v28.json'),h7=read('h7-v33.json'),h8=read('h8-attempt06.json');
  for(const [name,version]of [['h6-v28','h6'],['h7-v33','h7']]){
    const actual=read(name+'.json'),converted=E.quality(read('sources/'+name+'-quality.json'),version);
    for(const [key,v]of Object.entries(converted))assert.deepEqual(actual[key],v,'real artifact conversion '+key);
  }
  assert.deepEqual([h6.funnel.candidate,h6.funnel.teacher_proposed,h6.funnel.strict_accepted,h6.funnel.actor_query],[2035270,1102974,995127,995127]);
  assert.equal(D.value(h6.funnel.teacher_acceptance,true),'90.22%');assert.equal(D.value(h6.funnel.supervision_coverage,true),'48.89%');
  assert.deepEqual(['A_LOOSE_RECOVERY','B_CAGE_STOCK','CARRY_RETURN','CAGE_DELIVERY'].map(k=>h6.fragments[k].strict_accepted),[71549,371152,540012,12414]);
  assert.equal(h6.funnel.noneligible_gradient_pollution,0);assert(h6.accepted_is_query_lower_bound);
  const source=read('sources/h6-v28-quality.json');delete source.funnel.teacher_proposed;delete source.teacher_quality.legal_action_rate;
  assert.equal(E.quality(source,'h6').funnel.teacher_proposed,null);assert.equal(E.quality(source,'h6').teacher_quality.legal_action_rate,null);
  const projection=read('sources/h8-attempt06-counters.json'),funnel=read('sources/h8-attempt06-funnel.json');
  for(const [key,v]of Object.entries(E.smoke(projection,funnel)))assert.deepEqual(h8[key],v);
  assert.deepEqual(Object.values(h8.smoke_a),[50,44,0,0,2,26]);assert.equal(h8.scope,'smoke');assert.equal(h8.validation.passed,false);
  assert.equal(h8.validation.original_passed,true);assert.equal(h8.funnel.actor_query,296,'full smoke query is distinct from target A query 26');
  assert(D.FRAGMENTS.A_LOOSE_RECOVERY.includes('来源未区分'));
  const rcat=JSON.parse(fs.readFileSync('reports/review_catalog_before_gate_20261018.json'));
  for(const row of rcat.replays){
    if(row.id.startsWith('h7-'))assert.equal(row.checkpoint_sha256,h7.checkpoint_sha256);const replay=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
    assert.equal(replay.dagger_events,undefined);assert.equal(D.episodeStatus(replay),'本局逐tick DAgger：未记录');
    assert.deepEqual(D.atTick(replay,replay.events[0]?.tick??0),[],'never infer DAgger from key events/reward');
    assert(Object.values(D.episodeCounts(replay)).every(v=>v===null));
  }
  const event={tick:123,player:0,fragment:'A_ENEMY_DROP_RECOVERY',candidate:true,explicit_localization:true,
    teacher_proposed:true,strict_accepted:true,actor_query:true,actor_gradient:true,rejection_reasons:[],action_set:[[2,0],[4,1]]};
  D.validateEvents([event],2400);D.validateEvents([{tick:0,player:1,candidate:null}],2400);D.validateEvents([],2400);
  for(const mutate of [e=>e.tick=1.5,e=>e.tick=2401,e=>e.tick=-1,e=>e.player=2,e=>e.fragment='INFERRED',
    e=>e.candidate=1,e=>e.explicit_localization='yes',e=>e.teacher_proposed=false,e=>e.strict_accepted=false,
    e=>e.actor_query=false,e=>e.actor_gradient='yes',e=>e.rejection_reasons=['DOOMED'],e=>e.action_set=[[5,0]],
    e=>e.action_set=[[0,3]],e=>e.action_set=[[0,0],[0,0]],e=>e.action_set=[],e=>e.private_state={pos:[]}]){
    const bad=structuredClone(event);mutate(bad);assert.throws(()=>D.validateEvents([bad],2400),/schema/);
  }
  for(const events of [null,{},[event,event],[event,{...event,tick:122}],[{tick:1,player:0}],
    [{tick:1,player:0,rejection_reasons:['A','A']}],[{tick:1,player:0,rejection_reasons:['private / path']}]])assert.throws(()=>D.validateEvents(events,2400));
  const sparse={dagger_events:[event]};assert.deepEqual(D.atTick(sparse,124),[]);assert.deepEqual(D.atTick(sparse,123),[event]);
  assert.equal(D.episodeCounts({dagger_events:[{tick:1,player:0,candidate:false}]}).candidate.positive,0);
  const replay=JSON.parse(fs.readFileSync('web/review-replays/'+rcat.replays[0].file));replay.dagger_events=[event];P.validate(replay);
  const oldIdentity=replay.meta.replay_identity;replay.meta.replay_identity=await P.hash(P.identity(replay));assert.notEqual(replay.meta.replay_identity,oldIdentity);
  const level=require('../scripts/eval_web_hunter').loadLevel();await new P.Player(replay,level).verify();
  replay.dagger_events[0].action_set=[[0,0]];await assert.rejects(()=>new P.Player(replay,level).verify(),/identity hash/);
  console.log('DAgger GREEN: real H6/H7/H8 artifact hashes and counters, null preservation, own/enemy separation, sparse schema RED/GREEN, no inference, signal identity hash');
})().catch(e=>{console.error(e);process.exitCode=1;});
