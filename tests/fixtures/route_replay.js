'use strict';
// Synthetic protocol tests only. Never add this document to a public catalog.
const fs=require('fs');
module.exports=function fixture(){
  const catalog=require('../../web/review-catalog.json');
  const replay=JSON.parse(fs.readFileSync('web/review-replays/'+catalog.replays[0].file));
  const h='a'.repeat(64);
  const meta={schema:'qqt.route-potential/v1',kind:'fixture',status:'recorded',gamma:.995,lambda:.05,
    checkpoint_sha256:replay.meta.checkpoint_sha256,policy_sha256:replay.meta.model_sha256,
    simulator_sha256:replay.meta.simulator_sha256,reward_ledger_sha256:h,route_implementation_sha256:h,
    source_sha256:h,recomputed_sha256:h,source_run:'synthetic-fixture',generated_at:'2026-10-17T00:00:00Z',
    distance_metric:'production_weighted_grid_v1',shaped_critic:null};
  const row=(tick,extra={})=>({tick,route_phase:'C_RETURN',target_type:'home_cage',target_provenance:'held_enemy_bun',target_id:'fixture-object-1',
    phi_before:-12,phi_after:-11,route_potential_delta:1.055,route_shaped_reward:.05275,
    distance_before:12,distance_after:11,topology_region_before:'outer',topology_region_after:'inner',topology_gain:0,
    phase_transition:false,invalidated:false,terminal_correction:false,terminal:false,safety_reject_reason:null,...extra});
  const events=[row(1),row(2,{phi_before:-11,phi_after:-12,route_potential_delta:-.94,route_shaped_reward:-.047,distance_before:11,distance_after:12}),
    row(4,{route_phase:'D_DELIVERY',phi_before:-1,phi_after:-1,route_potential_delta:.005,route_shaped_reward:.00025,phase_transition:true,distance_before:'INF',distance_after:0,topology_gain:1,topology_evidence:[h,h,h]}),
    row(replay.actions.length,{phi_before:-1,phi_after:0,route_potential_delta:1,route_shaped_reward:.05,distance_before:0,distance_after:0,terminal:true,terminal_correction:true})];
  const rewardTick=replay.rewards.find(r=>r[0]>4&&r[0]<replay.actions.length)?.[0];
  if(rewardTick)events.push(row(rewardTick));events.sort((a,b)=>a.tick-b.tick);
  return {replay,source:{schema:'qqt.route-export-input/v1',replay_identity:replay.meta.replay_identity,meta,events,shaped_results:[]}};
};
