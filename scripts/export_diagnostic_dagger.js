'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),zlib=require('zlib'),assert=require('assert');
const D=require('../web/review_dagger');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const number=v=>Number.isFinite(v)?v:null;
const ratio=(a,b)=>a!==null&&b>0?a/b:null;
const reasons=['strict_verifier_reject','strict_verifier_reject_upper_bound','AMBIGUOUS','UNREACHABLE','DOOMED','UNCLEAR_PROVENANCE','FUTURE_DEPENDENT'];
const fragments={a_loose:'A_LOOSE_RECOVERY',b_stock:'B_CAGE_STOCK',carry_return:'CARRY_RETURN',cage_delivery:'CAGE_DELIVERY'};
const iso=v=>Number.isFinite(v)?new Date(v*1000).toISOString():null;
function quality(source,version){
  const old=version==='h6',f=source.funnel,t=source.teacher_quality;
  const per=old?source.fragments.per_fragment:source.fragments;
  const d={funnel:{candidate:number(old?f.candidate_fragments:f.candidates),
    explicit_localization:number(old?f.clearly_located:f.explicit_localization),
    teacher_proposed:number(old?f.teacher_proposed:f.proposed),strict_accepted:number(f.strict_accepted),
    actor_query:number(old?f.gradient_queries:f.gradient_query),teacher_acceptance:number(f.teacher_acceptance),
    supervision_coverage:number(old?f.final_supervision_coverage:f.supervision_coverage),
    noneligible_gradient_pollution:number(f.noneligible_gradient_pollution)},fragments:{},
    teacher_quality:{legal_action_rate:number(old?t.legal_action_rate:t.legal_query_rate),
      counterfactual_survival:number(old?t.rollout_survival_rate:t.counterfactual_survival),
      task_progress:number(old?t.task_progress_rate:t.task_progress),
      drop_expert_consistency:number(old?t.old_drop_expert_consistency:t.drop_expert_consistency),
      return_expert_consistency:number(old?t.old_return_expert_consistency:t.return_expert_consistency)},
    refusal_reasons:Object.fromEntries(reasons.map(k=>[k,number(t.refusal_reason_breakdown?.[k])])),
    action_set_size_histogram:t.action_set_size_histogram||null,
    accepted_is_query_lower_bound:old&&f.accepted_count_is_query_lower_bound===true};
  for(const [key,machine]of Object.entries(fragments)){const s=per[key]||{};
    d.fragments[machine]={candidate:number(s.candidate_ticks),teacher_proposed:number(s.proposed),
      strict_accepted:number(s.accepted),actor_query:number(old?s.queries:s.gradient_query)};}
  d.fragments.COMBAT_CONTROL={candidate:null,teacher_proposed:null,strict_accepted:null,
    actor_query:number(old?source.fragments.combat_control?.dagger_queries:source.combat_control?.teacher_queries)};
  return d;
}
function smoke(projection,funnel){
  const rows=projection.rows;
  const total=k=>rows.every(r=>Number.isFinite(r.counters[k]))?rows.reduce((n,r)=>n+r.counters[k],0):null;
  const candidate=total('teacher_eligible'),proposed=total('teacher_proposed'),accepted=total('teacher_accepted'),query=total('gradient_query');
  const d={funnel:{candidate,explicit_localization:null,teacher_proposed:proposed,strict_accepted:accepted,
    actor_query:query,teacher_acceptance:ratio(accepted,proposed),supervision_coverage:ratio(query,candidate),
    noneligible_gradient_pollution:total('noneligible_label')},fragments:{},
    teacher_quality:{legal_action_rate:ratio(total('legal_query'),query),counterfactual_survival:null,
      task_progress:null,drop_expert_consistency:null,return_expert_consistency:null},
    refusal_reasons:Object.fromEntries(reasons.map(k=>[k,null])),
    action_set_size_histogram:Object.fromEntries([1,2,3,4,5].map(n=>[n,total('action_set_size_'+n)])),
    accepted_is_query_lower_bound:false};
  for(const [key,machine]of Object.entries(fragments))d.fragments[machine]={candidate:total(key+'_candidate_ticks'),
    teacher_proposed:total(key+'_proposed'),strict_accepted:total(key+'_accepted'),actor_query:total(key+'_query')};
  d.fragments.COMBAT_CONTROL={candidate:null,teacher_proposed:null,strict_accepted:null,actor_query:total('combat_control_dagger_query')};
  d.smoke_a={stock_theft:total('target_a_courier_stock_theft'),enemy_carried_drop:total('target_a_enemy_carried_drop'),
    enemy_origin_pickup:total('target_a_enemy_origin_pickup'),enemy_path_query:total('path_a_enemy'),
    own_recovery_pickup:total('target_a_own_recovery_pickup'),target_a_query:total('target_a_query')};
  for(const [k,v]of Object.entries({stock_theft:'stock_theft',enemy_carried_drop:'drop_spawned',enemy_origin_pickup:'enemy_origin_pickup',own_recovery_pickup:'own_recovery_pickup',target_a_query:'actor_query'})){
    assert.equal(d.smoke_a[k],funnel.funnel[v],`H8 source disagreement: ${k}`);
  }
  d.validation={passed:false,original_passed:funnel.passed,strict_declared_acceptance:funnel.strict_declared_acceptance,
    failed_requirements:funnel.failed_requirements};
  assert.equal(d.smoke_a.enemy_path_query,0);assert.equal(d.smoke_a.enemy_origin_pickup,0);
  assert.equal(d.validation.strict_declared_acceptance,false);
  return d;
}
function exportAll(root,out='web/review-dagger'){
  assert(root,'source root required');const generated_at=new Date().toISOString();
  fs.mkdirSync(path.join(out,'sources'),{recursive:true});
  const paths={h6:'runs/je09_bun_20261016/stage_h6_full_trajectory_fragment_dagger_v28',
    h7:'runs/je09_bun_20261017/stage_h7_native_courier_fsm_dagger_v33',
    h8:'runs/je09_bun_20261017/stage_h8_native_courier_fsm_dagger_v34'};
  function save(file,bytes){
    // Only public relative provenance and structured numeric artifacts are exported.
    const s=bytes.toString();assert(!/\/mnt\/|\/home\/|\/afs\/|\b(?:10|192\.168)\.\d+\.\d+\.\d+\b/.test(s),'private source content');
    fs.writeFileSync(path.join(out,file),bytes);return {file,bytes:bytes.length,sha256:sha(bytes),gzip_bytes:zlib.gzipSync(bytes,{level:9}).length};
  }
  function read(version,suffix,file){const relative=paths[version]+'/'+suffix,bytes=fs.readFileSync(path.join(root,relative)),doc=JSON.parse(bytes);
    const asset=save('sources/'+file,bytes);return {doc,source:{file:asset.file.slice(8),path:relative,schema:doc.schema??null,sha256:asset.sha256,generated_at:iso(doc.generated_unix??doc.unix)}};}
  const h6=read('h6','quality/DAGGER_QUALITY.json','h6-v28-quality.json');
  const h7=read('h7','quality/DAGGER_QUALITY_LIVE.json','h7-v33-quality.json');
  const committed=read('h7','formal/bun392_bridge_pfsp8/gen000032/COMMITTED.json','h7-v33-committed.json');
  const h8=read('h8','state/H8_SMOKE_A_FUNNEL.json','h8-attempt06-funnel.json');
  assert.equal(h6.doc.schema,'je09.h6.dagger-quality/v1');assert.equal(h7.doc.schema,'je09.h7.dagger-quality-live/v1');
  assert(committed.doc.passed&&committed.doc.generation===32&&h7.doc.generations.includes(32));
  const relative=paths.h8+'/production_smoke_stage_h8_v34_attempt06/formal/bun392_bridge_pfsp8/events.jsonl';
  const bytes=fs.readFileSync(path.join(root,relative));
  const counters=['teacher_eligible','teacher_proposed','teacher_accepted','gradient_query','legal_query','noneligible_label','combat_control_dagger_query',
    'target_a_courier_stock_theft','target_a_enemy_carried_drop','target_a_enemy_origin_pickup','target_a_own_recovery_pickup','path_a_enemy',
    ...Object.keys(fragments).flatMap(k=>['candidate_ticks','proposed','accepted','query'].map(s=>k+'_'+s)),
    ...[1,2,3,4,5].map(n=>'action_set_size_'+n)];
  const projection={schema:'qqt.dagger-smoke-counters/v1',source:{run:paths.h8,path:relative,schema:null,sha256:sha(bytes)},
    rows:bytes.toString().trim().split('\n').map(line=>{const e=JSON.parse(line);return {generation:e.generation,unix:e.unix,
      counters:{...Object.fromEntries(counters.map(k=>[k,number(e.dagger?.[k])])),target_a_query:number(e.dagger?.targeted_course?.['1']?.query)}};})};
  assert.deepEqual(projection.rows.map(r=>r.generation),[1,2,3,4]);
  const projected=save('sources/h8-attempt06-counters.json',Buffer.from(JSON.stringify(projection,null,2)+'\n'));
  const data=[{id:'h7-v33',title:'H7 v33 · 本模型训练汇总',scope:'run',source_run:paths.h7,
    checkpoint_sha256:committed.doc.policy_sha256,generations:h7.doc.generations,source_scopes:h7.doc.source_runs,
    sources:[h7.source,committed.source],source_generated_at:iso(h7.doc.generated_unix),...quality(h7.doc,'h7')},
    {id:'h8-attempt06',title:'H8 v34 attempt06 · Smoke 严格验收失败',scope:'smoke',source_run:paths.h8,
    checkpoint_sha256:null,generations:[1,2,3,4],source_scopes:['smoke'],sources:[h8.source,
      {file:'h8-attempt06-counters.json',path:relative,schema:projection.schema,sha256:projected.sha256,
        original_sha256:projection.source.sha256,generated_at:iso(projection.rows.at(-1).unix)}],
    source_generated_at:iso(projection.rows.at(-1).unix),...smoke(projection,h8.doc)},
    {id:'h6-v28',title:'H6 v28 · 历史参考',scope:'historical',source_run:paths.h6,checkpoint_sha256:null,
      generations:h6.doc.runtime_scope.generations,source_scopes:['smoke','formal'],sources:[h6.source],source_generated_at:iso(h6.doc.generated_unix),...quality(h6.doc,'h6')}];
  const runs=data.map(row=>{const doc=D.validateQuality({schema:D.SCHEMA,generated_at,...row});const asset=save(row.id+'.json',Buffer.from(JSON.stringify(doc,null,2)+'\n'));
    return {id:row.id,title:row.title,scope:row.scope,checkpoint_sha256:row.checkpoint_sha256,...asset};});
  const catalog={schema:'qqt.dagger-catalog/v1',generated_at,runs};save('catalog.json',Buffer.from(JSON.stringify(catalog,null,2)+'\n'));
  return catalog;
}
module.exports={quality,smoke,exportAll};
if(require.main===module){const catalog=exportAll(process.argv[2]);console.log(JSON.stringify(catalog));}
