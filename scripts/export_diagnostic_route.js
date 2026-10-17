'use strict';
// Read-only adapter: no training imports, inference, GPU access or catalog writes.
const fs=require('fs'),path=require('path'),zlib=require('zlib');
const P=require('../web/review_protocol'),A=require('../web/review_advantage'),R=require('../web/review_route');
const INPUT='qqt.route-export-input/v1';
const quant=v=>v==null?null:Math.round(v/1e-6);
function checkRows(source,replay){
  if(Object.keys(source).some(k=>!['schema','replay_identity','meta','events','shaped_results','task_results','task_advantage'].includes(k))||source.schema!==INPUT||source.replay_identity!==replay.meta.replay_identity||!Array.isArray(source.events)||!Array.isArray(source.shaped_results))throw Error('route input binding');
  const allowed=['tick','target_type','target_provenance','target_id',...R.COLUMNS.filter(k=>!['ticks','target','shaped_legal_count'].includes(k)&&!k.startsWith('shaped_'))];
  for(const row of source.events)if(!row||Object.keys(row).some(k=>!allowed.includes(k)))throw Error('route unexpected input column');
}
function decision(row,replay,gamma){
  if(!Array.isArray(row.taken)||row.taken.length!==2||JSON.stringify(row.taken)!==JSON.stringify(replay.actions[row.tick-1].slice(0,2)))throw Error('route decision action');
  const candidates=row.candidates;
  if(!Array.isArray(candidates)||candidates.some(c=>!c||!Array.isArray(c.action)||c.action.length!==2||typeof c.legal!=='boolean'||typeof c.terminal!=='boolean')||
    candidates.filter(c=>c.legal).some(c=>!Number.isInteger(c.action[0])||c.action[0]<0||c.action[0]>4||!Number.isInteger(c.action[1])||c.action[1]<0||c.action[1]>2)||
    new Set(candidates.map(c=>c.action.join(','))).size!==candidates.length)throw Error('route joint candidates');
  return A.calculate(candidates,row.taken,gamma);
}
async function compile(replay,source){
  checkRows(source,replay);
  const m=structuredClone(source.meta),d={schema:R.SCHEMA,scale:1e-6,targets:[]};
  for(const k of R.COLUMNS)d[k]=[];
  const results=new Map();let previous=null,last=0;
  for(const r of source.shaped_results){
    if(!m.shaped_critic||!Number.isInteger(r.tick)||r.tick<=last||r.tick>replay.actions.length||JSON.stringify(r.taken)!==JSON.stringify(replay.actions[r.tick-1].slice(0,2)))throw Error('route shaped result / action binding');
    last=r.tick;
    const result=decision(r,replay,m.gamma);
    results.set(r.tick,{shaped_critic_q1_taken:result.taken_q,shaped_critic_q1_policy_mean:result.mean_q,shaped_advantage_a1:result.advantage,
      shaped_advantage_delta:previous&&r.tick===previous.tick+1?result.advantage-previous.a:null,shaped_legal_count:r.candidates.filter(c=>c.legal).length});
    previous={tick:r.tick,a:result.advantage};
  }
  const events=new Map();last=0;
  for(const e of source.events){if(!Number.isInteger(e.tick)||e.tick<=last||e.tick>replay.actions.length)throw Error('route event order');last=e.tick;events.set(e.tick,e);}
  for(const tick of [...new Set([...events.keys(),...results.keys()])].sort((a,b)=>a-b)){
    const e=events.get(tick)||{},r=results.get(tick)||{};d.ticks.push(tick);
    let target=null;
    if(e.target_id!=null){
      if(typeof e.target_id!=='string'||e.target_id.length>1024)throw Error('route target identity');
      const id='target-'+await P.hash([replay.meta.id,replay.meta.seed,e.target_type,e.target_provenance,e.target_id]);
      target=d.targets.findIndex(t=>t.id===id);
      if(target<0){target=d.targets.length;d.targets.push({type:e.target_type,provenance:e.target_provenance,id});}
    }else if(e.target_type!=null||e.target_provenance!=null)throw Error('route target incomplete');
    d.target.push(target);
    for(const k of R.COLUMNS.filter(k=>!['ticks','target'].includes(k))){const v=k.startsWith('shaped_')?r[k]:e[k];
      if(R.VALUES.includes(k)&&v!=null&&!Number.isFinite(v))throw Error('route finite quantity');
      d[k].push(R.VALUES.includes(k)?quant(v):Array.isArray(v)?v.slice():v??null);
    }
  }
  m.measured_count=d.route_potential_delta.filter(v=>v!==null).length;m.missing_count=replay.actions.length-m.measured_count;
  m.shaped_measured_count=results.size;m.shaped_missing_count=replay.actions.length-results.size;
  let task=null;
  if(source.task_results!==undefined){
    if(!Array.isArray(source.task_results)||!source.task_advantage)throw Error('route original task evidence');
    const data={...A.empty(),legal_count:[]},meta=structuredClone(source.task_advantage);let prior=null;
    const allowed=['schema','status','reason','definition','gamma','action_scope','legal_actions','probability_source','checkpoint_sha256','policy_sha256','critic_sha256','simulator_sha256','reward_hashes','counterfactual_sha256','measured_count','missing_count','missing_reasons','quantization_error','calculation_version','rng_status','critic_network_sha256','input_encoding','calculation_hashes'];
    if(Object.keys(meta).some(k=>!allowed.includes(k)))throw Error('route original task metadata');
    for(const row of source.task_results){
      if(!Number.isInteger(row.tick)||row.tick<1||row.tick>replay.actions.length||prior&&row.tick<=prior.tick||JSON.stringify(row.taken)!==JSON.stringify(replay.actions[row.tick-1].slice(0,2)))throw Error('route original task binding');
      const result=decision(row,replay,meta.gamma);
      data.ticks.push(row.tick);data.taken_q.push(quant(result.taken_q));data.mean_q.push(quant(result.mean_q));data.advantage.push(quant(result.advantage));data.delta.push(prior&&row.tick===prior.tick+1?quant(result.advantage-prior.a):null);data.legal_count.push(row.candidates.filter(c=>c.legal).length);
      prior={tick:row.tick,a:result.advantage};
    }
    meta.measured_count=data.ticks.length;meta.missing_count=replay.actions.length-data.ticks.length;
    A.validate({...replay,meta:{...replay.meta,advantage:meta},advantages:data});
    if(replay.meta.advantage?.status==='recorded'){
      if(meta.critic_sha256!==replay.meta.advantage.critic_sha256||meta.gamma!==replay.meta.advantage.gamma)throw Error('route original task critic differs');
      for(const tick of data.ticks){const i=data.ticks.indexOf(tick),old=A.atTick(replay,tick);for(const k of ['taken_q','mean_q','advantage','delta']){if(k==='delta'&&data[k][i]===null)continue;if(old[k]===null?data[k][i]!==null:data[k][i]===null||Math.abs(old[k]-data[k][i]*data.scale)>2e-6)throw Error('route original task mismatch');}}
      // Verification of raw results may be sparse; existing original measurements never change.
    }else task={meta,data};
  }else if(source.task_advantage!==undefined)throw Error('route original task results absent');
  return {meta:m,data:d,task};
}
async function convert(replay,source,recomputed){
  P.validate(replay);
  if(await P.hash(P.identity(replay))!==replay.meta.replay_identity)throw Error('route input replay identity');
  const one=await compile(replay,source),two=await compile(replay,recomputed);
  if(source.meta.kind==='production'&&source===recomputed)throw Error('route separate independent recomputation required');
  if(JSON.stringify(one.data)!==JSON.stringify(two.data)||JSON.stringify(one.meta)!==JSON.stringify(two.meta)||JSON.stringify(one.task)!==JSON.stringify(two.task))throw Error('route independent recompute mismatch');
  const document=structuredClone(replay);document.meta.route=one.meta;document.route=one.data;
  if(one.task){document.meta.advantage=one.task.meta;document.advantages=one.task.data;}
  document.meta.route.source_sha256=await P.hash(source);document.meta.route.recomputed_sha256=await P.hash(recomputed);
  document.meta.replay_identity=await P.hash(P.identity(document));P.validate(document);
  const bytes=Buffer.from(JSON.stringify(document)+'\n'),routeBytes=Buffer.from(JSON.stringify(one.data));
  return {document,report:{schema:'qqt.route-export-report/v1',kind:one.meta.kind,replay_id:document.meta.id,
    replay_identity:document.meta.replay_identity,sha256:await P.sha256(bytes),raw_bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes,{level:9}).length,
    route_raw_bytes:routeBytes.length,route_gzip_bytes:zlib.gzipSync(routeBytes,{level:9}).length,
    measured_count:one.meta.measured_count,missing_count:one.meta.missing_count,shaped_measured_count:one.meta.shaped_measured_count,
    source_sha256:document.meta.route.source_sha256,recomputed_sha256:document.meta.route.recomputed_sha256}};
}
module.exports={INPUT,convert};
if(require.main===module)(async()=>{
  const args=process.argv.slice(2),opts={};for(let i=0;i<args.length;i+=2)opts[args[i]]=args[i+1];
  if(['--replay','--input','--recomputed','--output','--report'].some(k=>!opts[k]))throw Error('Required: --replay --input --recomputed --output --report');
  const read=k=>JSON.parse(fs.readFileSync(opts[k])),source=read('--input');
  // Never write test fixtures into the deployable tree, and never rewrite a catalog.
  if([opts['--output'],opts['--report']].some(p=>/catalog/i.test(path.basename(p))||source.meta.kind==='fixture'&&path.resolve(p).startsWith(path.resolve('web')+path.sep)))throw Error('route fixture / catalog publication forbidden');
  const replay=read('--replay');
  if(source.meta.kind==='production'){
    if(fs.realpathSync(opts['--input'])===fs.realpathSync(opts['--recomputed']))throw Error('route independent input required');
    await new P.Player(replay,require('./eval_web_hunter').loadLevel()).verify();
  }
  const {document,report}=await convert(replay,source,read('--recomputed'));
  fs.writeFileSync(opts['--output'],JSON.stringify(document)+'\n',{flag:'wx'});fs.writeFileSync(opts['--report'],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(report));
})().catch(()=>{console.error('Route export rejected: check schema, bindings and independent recomputation; no catalog was changed.');process.exitCode=1;});
