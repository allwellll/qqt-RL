'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const B=process.env.BT_REPLAY_BASE||'runs/v2_fresh_selection_20261031/replay',P=require('../../cx_publish_r1_1805_20261030/web/review_protocol');
const manifest=JSON.parse(fs.readFileSync(B+'/MANIFEST.json')),level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const final=B.endsWith('/final_replay');
(async()=>{
 const rows=[];
 for(const entry of manifest.games){
  const doc=JSON.parse(fs.readFileSync(`${B}/game_${String(entry.index).padStart(3,'0')}/review.json`)),player=new P.Player(doc,level);await player.verify();player.seek(0);const s=player.sim,p=entry.seat;
  const events=[],positions=[],carry=[],bubbleReq=[],itemReq=[];
  for(let t=0;t<doc.actions.length;t++){
   const before={carry:s.bunCarried[p],stored:s.bunStored.map(a=>a.slice()),alive:s.alive[p],trap:s.trapped[p]};
   const a=doc.actions[t];if(a[2*p+1]===1)bubbleReq.push(t+1);if(a[2*p+1]===2)itemReq.push(t+1);
   player.step();positions.push([s.pos[2*p],s.pos[2*p+1]]);carry.push(s.bunCarried[p]>=0);
   if(!before.trap&&s.trapped[p])events.push({tick:s.t,event:'controlled'});
   if(before.alive&&!s.alive[p])events.push({tick:s.t,event:'death'});
   if(before.carry<0&&s.bunCarried[p]>=0)events.push({tick:s.t,event:'pickup',origin:s.bunCarried[p]});
   if(before.carry>=0&&s.bunCarried[p]<0)events.push({tick:s.t,event:'carry_end',stored_increased:s.bunStored.some((a,i)=>a.some((v,j)=>v>before.stored[i][j]))});
  }
  const plateaus=[];let start=null;
  for(let i=1;i<positions.length;i++){
   const fixed=carry[i]&&carry[i-1]&&Math.hypot(...positions[i].map((v,k)=>v-positions[i-1][k]))<1e-5;
   if(fixed&&start===null)start=i-1;
   if((!fixed||i===positions.length-1)&&start!==null){if(i-start>=20)plateaus.push({start_tick:start+1,end_tick:i+1,ticks:i-start,scope:'carrying and position unchanged; may be controlled/blocked/intentional, cause not inferred'});start=null;}
  }
  assert(s.done);
  rows.push({model:entry.model||manifest.model,seed:entry.seed,seat:p,winner:s.winner,ticks:s.t,selection:entry.rule,bomb_request_ticks:bubbleReq,item_request_ticks:itemReq,events,carry_stationary_intervals:plateaus});
 }
 fs.writeFileSync(B+'/OBSERVABLE_TIMELINE.json',JSON.stringify({status:'COMPLETE',rule_version:'bun-native-v2',scope:final?'fixed first8 and disclosed earliest outcomes; observed events, not representative frequencies or inferred causes':'six fixed P0 first games only; observed events, no policy Q or inferred cause',rows},null,2));
 const lines=['# v2录像事实时间线（20261031）','',final?'固定首8局，另披露选择各席最早2胜/1负及P1最早单方零泡；重复入口不增加独立样本数。选局用于观察短板，不估计胜负频率。':'这是六模型各自固定hard首局、全部P0的预览，不按胜负选取。它不证明P1改善，也不能代表完整6144局。',`时间=tick/10秒；放泡/道具请求与实际放置分开。完整原始Review/hash/在线播放证据在 \`${B}/\`。`,''];
 for(const r of rows){lines.push(`## ${r.model.checkpoint} · seed${r.seed} · P${r.seat}`,`选局：${r.selection}；终局tick${r.ticks}，winner=${r.winner}；policy SHA ${r.model.policy_sha256}。`, `首次放泡请求 ${r.bomb_request_ticks[0]??'无'} tick；道具请求 ${r.item_request_ticks.length} 次。`,...r.events.map(e=>`- ${(e.tick/10).toFixed(1)}秒（tick${e.tick}）：${JSON.stringify(e)}`),...r.carry_stationary_intervals.map(e=>`- 持包位置不变 ${(e.start_tick/10).toFixed(1)}–${(e.end_tick/10).toFixed(1)}秒；${e.ticks}tick；只记录事实，不猜测受控/碰撞/主动等待原因。`),'');}
 fs.writeFileSync(process.env.V2_REPLAY_REPORT||(final?'reports/v2_winner_replay_observations_20261031.md':'reports/v2_replay_observations_20261031.md'),lines.join('\n'));
 console.log('full Review observed timelines PASS',rows.length);
})().catch(e=>{console.error(e);process.exitCode=1});
