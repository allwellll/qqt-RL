'use strict';
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict'),zlib=require('node:zlib');
const P=require('../web/review_protocol'),R=require('../web/review_route'),G=require('./generate_v36_multiseed_review');
const hash=v=>crypto.createHash('sha256').update(typeof v==='string'||Buffer.isBuffer(v)?v:JSON.stringify(v)).digest('hex');
const PHASE=['NONE','B_STOCK','A_ENEMY_DROP','C_RETURN','D_DELIVERY','PRESTOCK_HOME','PRESTOCK_ENEMY'];
const TARGET={B_STOCK:['enemy_cage','enemy_stock'],A_ENEMY_DROP:['loose_bun','enemy_carried_drop'],C_RETURN:['home_cage','held_enemy_bun'],D_DELIVERY:['home_cage','held_enemy_bun'],PRESTOCK_HOME:['route_frontier','home_corridor_preparation'],PRESTOCK_ENEMY:['route_frontier','stock_corridor_preparation']};
function columns(rows,seed,beforeStates){
  const out={schema:R.SCHEMA,scale:1e-6,targets:[]},ids=new Map();for(const k of R.COLUMNS)out[k]=[];
  for(const [index,row] of rows.entries()){
    const m=row.route[0],phase=PHASE[m.route_phase];let target=null;
    if(!phase)throw Error('unknown route phase');
    if(phase==='D_DELIVERY'){
      const s=beforeStates?.[index],base=s?.bunBases?.[0],y=s?.pos?.[0],x=s?.pos?.[1];
      if(!base||s.bunCarried[0]!==1||!(y>=base[0]&&y<base[0]+3&&x>=base[1]&&x<base[1]+3))throw Error('route phase D_DELIVERY requires actual home cage target evidence; HOME half is not cage');
    }
    if(phase!=='NONE'){
      const key=phase+':'+m.route_target_identity;
      if(!ids.has(key)){ids.set(key,out.targets.length);const [type,provenance]=TARGET[phase];out.targets.push({type,provenance,id:'target-'+hash({seed,key})});}target=ids.get(key);
    }
    const values={ticks:row.tick,route_phase:phase,target,distance_before:phase==='NONE'?null:m.route_distance_before===32767?'INF':m.route_distance_before,distance_after:phase==='NONE'?null:m.route_distance_after===32767?'INF':m.route_distance_after,phi_before:m.route_phi_before,phi_after:m.route_phi_after,route_potential_delta:m.route_potential_delta,route_shaped_reward:m.route_shaped_reward,phase_transition:Boolean(m.route_phase_transition),invalidated:Boolean(m.route_invalidated),terminal_correction:Boolean(m.route_terminal_correction),terminal:row.tick===rows.length};
    // Topology gain is a current-ledger statistic, not additional reward. Full
    // branch-specific safety provenance is not published as an exact certificate.
    for(const k of R.COLUMNS){let value=values[k]??null;if(value!==null&&R.VALUES.includes(k))value=Math.round(value/out.scale);out[k].push(value);}
  }
  return out;
}
async function main(){
  const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json'));
  for(const m of G.MATRIX){
    const id=G.idFor(m),file='web/review-replays/'+id+'.json',doc=JSON.parse(fs.readFileSync(file)),rewardPath='runs/replay_review_20261017/reward-traces/'+id+'.reward.json',raw=fs.readFileSync(rewardPath),reward=JSON.parse(raw);
    assert.equal(reward.status,'offline_recomputed_current_reward');assert.equal(reward.ticks.length,doc.actions.length);
    const previousChecks=JSON.stringify(doc.checks),previousActions=JSON.stringify(doc.actions);
    doc.rewards=[];doc.summary.reward=[0,0];doc.events=doc.events.filter(e=>e.kind!=='approx_phi_discontinuity');
    const jumps=[];let previousPhi=null;
    for(let i=0;i<reward.ticks.length;i++){
      const tick=reward.ticks[i],sparse=tick.components.map(c=>Object.fromEntries(Object.entries(c).filter(([,v])=>v!==0)));
      if(sparse.some(c=>Object.keys(c).length))doc.rewards.push([tick.tick,...sparse]);
      for(let p=0;p<2;p++)doc.summary.reward[p]+=tick.total[p];
      const jump=previousPhi===null?null:tick.route[0].route_phi_before-previousPhi;
      if(jump!==null&&Math.abs(jump)>1e-6){jumps.push({tick:tick.tick,approx_phi_jump:jump});doc.events.push({tick:tick.tick,kind:'approx_phi_discontinuity',player:0,approx_phi_jump:jump});}
      doc.diagnostic_ticks[i].offline_recomputed_current_reward={status:reward.status,components:tick.components,approx_total_reward:tick.total,route:tick.route,approx_phi_jump:jump};
      previousPhi=tick.route[0].route_phi_after;
    }
    const proof={schema:reward.schema,status:reward.status,reward_source:reward.reward_source,assumptions:reward.assumptions,source_hashes:reward.source_hashes,component_names:reward.component_names,measured_count:reward.ticks.length,missing_count:0,reward_artifact_sha256:hash(raw),approx_value_status:'not_computed',approx_a1_status:'not_computed; no legal counterfactual reward/Critic branches evaluated',formula:'approx_total_reward=sum(current reward components); route_shaped_reward=lambda*(gamma*phi_after-phi_before), terminal phi_after=0; no strict A1 claim',checkpoint_sha256:doc.meta.checkpoint_sha256};
    proof.converter_sha256=hash(fs.readFileSync('scripts/finalize_v36_current_review.js'));
    proof.potential_discontinuities=jumps;proof.potential_continuity='raw current ledger; cross-tick discontinuities preserved as approx_phi_jump, never smoothed';
    doc.events.sort((a,b)=>a.tick-b.tick);
    doc.meta.offline_recomputed_current_reward=proof;doc.meta.reward_status=reward.status;doc.meta.reward_source='offline_recomputed_current_reward / 近似诊断；当前奖励逻辑在真实 v36 Web 轨迹冷启动重算，非 training-time reward';doc.meta.reward_components=reward.component_names;
    doc.route=columns(reward.ticks,m.seed);
    doc.meta.route={...doc.meta.route,generated_at:fs.statSync(rewardPath).mtime.toISOString(),source_sha256:hash(raw),recomputed_sha256:hash(reward.ticks),reward_ledger_sha256:reward.source_hashes['scripts/recompute_v36_current_reward.py'],route_implementation_sha256:reward.source_hashes['scripts/je09_route_pbrs.py'],measured_count:reward.ticks.length,missing_count:0,semantics:'diagnostic-post-target-uncorrected/v1',limitations:'offline_recomputed_current_reward / 近似诊断：当前生产路线逻辑作用于 Web 观测状态及冷启动 aux；终局 Phi 归零。非 training-time reward，非严格 A1。假设与当前源码/字节码 hash 见本局离线重算证据。'};
    doc.meta.advantage_audit.reason='当前 reward/路线已按 offline_recomputed_current_reward 近似重算；同 checkpoint 没有独立原始任务 Critic，也没有真实历史训练 ledger/aux 对齐和全合法反事实 Q 分支，原始/塑形严格 A1/ΔA 仍缺测。PBRS Critic 参数存在不等于已计算优势。';
    doc.meta.advantage_audit.simulator_sha256=doc.meta.simulator_sha256;
    doc.meta.advantage_audit.critic_status='v36 g32 PBRS Critic：75参数叶、8副本精确一致；未计算反事实Q/A';
    doc.meta.advantage.reason=doc.meta.advantage_audit.reason;
    doc.meta.diagnostic_ticks_sha256=hash(doc.diagnostic_ticks);doc.meta.replay_identity=await P.hash(P.identity(doc));P.validate(doc);
    assert.equal(JSON.stringify(doc.actions),previousActions);assert.equal(JSON.stringify(doc.checks),previousChecks);
    const bytes=Buffer.from(JSON.stringify(doc));fs.writeFileSync(file,bytes);
    const row=catalog.replays.find(r=>r.id===id);Object.assign(row,{summary:doc.summary,sha256:hash(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes,{level:9}).length});
    console.log(JSON.stringify({seed:m.seed,ticks:doc.actions.length,approx_reward:doc.summary.reward,route_ticks:doc.meta.route.measured_count,strict_a1_ticks:0}));
  }
  catalog.v36_fixed_eight.current_reward_status='offline_recomputed_current_reward';catalog.v36_fixed_eight.current_reward_coverage=1;catalog.v36_fixed_eight.route_coverage=1;catalog.v36_fixed_eight.critical_scope='Current logic approximate diagnosis; never training-time reward or strict A1';
  catalog.replays=[...G.MATRIX.map(m=>catalog.replays.find(r=>r.id===G.idFor(m))),...catalog.replays.filter(r=>r.batch_id!=='v36-fixed-eight-20261017')];
  catalog.raw_bytes=catalog.replays.reduce((n,r)=>n+r.bytes,0);catalog.gzip_bytes=catalog.replays.reduce((n,r)=>n+r.gzip_bytes,0);
  fs.writeFileSync('web/review-catalog.json',JSON.stringify(catalog,null,2)+'\n');
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={columns};
