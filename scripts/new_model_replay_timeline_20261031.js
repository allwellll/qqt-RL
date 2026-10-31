'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const B=process.env.NEW_REPLAY_BASE||'runs/new_model_replays_20261031',file=B+'/MANIFEST.json';
const cfg=JSON.parse(fs.readFileSync(file));
const P=require('../../cx_publish_r1_1805_20261030/web/review_protocol');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
(async()=>{
 const rows=[];
 for(const entry of cfg.games){
  const doc=JSON.parse(fs.readFileSync(`${B}/game_${String(entry.index).padStart(3,'0')}/review.json`));
  const player=new P.Player(doc,level);await player.verify();player.seek(0);
  const s=player.sim,p=entry.seat,enemy=1-p,annotations=[],observed=[];
  let placed=0;
  for(const action of doc.actions){
   const before={carry:s.bunCarried[p],stored:s.bunStored.map(x=>x.slice()),alive:Array.from(s.alive),trap:Array.from(s.trapped),held:s.heldItem[p]};
   player.step();const info=player.info;
   const add=label=>{const a={start:s.t,end:Math.min(s.t+1,doc.actions.length),label};annotations.push(a);observed.push({tick:s.t,seconds:s.t/10,label});};
   if(info.placed[p]&&placed++===0)add('模型首次实际放泡');
   for(let q=0;q<2;q++){
    const who=q===p?'模型':'猎手';
    if(!before.trap[q]&&s.trapped[q])add(who+'被原生包裹');
    if(before.alive[q]&&!s.alive[q])add(who+'死亡');
   }
   if(before.carry<0&&s.bunCarried[p]>=0)add('模型真实拾包，来源队'+s.bunCarried[p]);
   if(before.carry>=0&&s.bunCarried[p]<0)add(s.bunStored[p][before.carry]>before.stored[p][before.carry]?(before.carry===enemy?'模型敌源包送达':'模型己方来源包归还库存（非进攻送达）'):'模型持包结束／掉包，非送达');
  }
  assert.equal(s.done,true);assert.equal(placed,entry.expected.placed[p]);
  const r=entry.expected.r3;
  for(const c of r.chains.filter(c=>c.player===p))annotations.push({start:c.tick,end:Math.min(c.tick+1,s.t),label:`模型真连锁：触发${c.triggered.length}泡／${c.batch_count}批`});
  for(const c of r.contacts.filter(c=>c.player===p))annotations.push({start:c.tick,end:Math.min(c.tick+1,s.t),label:(c.type===1?'踩香蕉':'触胶')+(c.owner===p?'（自己放置）':'（对手放置）')+(c.carried>=0?'·持包':'·空手')});
  annotations.sort((a,b)=>a.start-b.start);
  entry.annotations=annotations;
  rows.push({index:entry.index,model:entry.model.display_name,seed:entry.seed,seat:p,selection:entry.rule,observed,annotations,
   result:s.winner==null||s.winner===-1?'平':s.winner===p?'胜':'负',ticks:s.t,
   deliveries:entry.expected.primary_delivery,causal_kills:entry.expected.causal_kills[p],placed,
   true_chain_events:r.chains.filter(c=>c.player===p).length,
   pickups:r.pickups[p],releases:r.releases[p],own_banana_carry_contacts:r.contacts.filter(c=>c.player===p&&c.type===1&&c.owner===p&&c.carried>=0).length});
 }
 fs.writeFileSync(file,JSON.stringify(cfg,null,2));
 fs.writeFileSync(B+'/OBSERVABLE_TIMELINE.json',JSON.stringify({status:'FULL_REVIEW_RECOMPUTED',rows,limitations:['Chinese timeline records observations, not Q/advantage or causal net item benefit','selected examples do not estimate frequencies']},null,2));
 console.log(JSON.stringify({status:'PASS',entries:rows.length}));
})().catch(e=>{console.error(e);process.exitCode=1});
