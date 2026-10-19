'use strict';
// Native web Bun transportation authority; separate from frozen Hunter arena.
const fs=require('fs');const assert=require('node:assert/strict');
const QQT=require('../web/sim.js');const authority=require('./eval_web_hunter.js');
const inference=require('./je05_web_hunter.js');const residual=require('./eval_web_hunter_residual.js');
const tasks=['carry_home','near_steal','route_break','full'];
function setup(task,seed){
  const s=new QQT.Sim(seed);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
  s.maxSteps=300;s.initialHp=1;s.hp=[1,1];s.invuln=[0,0];
  if(task==='carry_home'){
    s.brick.fill(0);s.pos[0]=9.5;s.pos[1]=4.5+(seed%3);s.bunCarried[0]=1;s.bunStored[1][1]=0;
  }else if(task==='near_steal'){
    // One legal walkable cell outside enemy cage: no idle first-tick theft.
    const cells=[];
    for(let r=0;r<QQT.H;r++)for(let c=0;c<QQT.W;c++){
      const i=r*QQT.W+c;if(s.wall[i]||s.brick[i]||s._bunBaseTeam(r,c)>=0)continue;
      const b=s.bunBases[1];const d=Math.max(b[0]-r,0,r-(b[0]+2))+Math.max(b[1]-c,0,c-(b[1]+2));
      if(d===1)cells.push([r,c]);
    }
    assert.ok(cells.length);const [r,c]=cells[seed%cells.length];s.pos[0]=r+.5;s.pos[1]=c+.5;s.trapped[1]=30+seed%30;
  }else if(task==='route_break'){
    s.pos[0]=9.5;s.pos[1]=2.5;
  }
  s.bunSpawnPos[0]=[s.pos[0],s.pos[1]];s._gen++;
  return s;
}
function defaultLedger(){return {high:-1,seen:false,marks:0};}
function updateFunnel(s,b,info,ledger,table){
  const out={approach:0,threat:0,sugar:0,kill:0,steal_trap:0,steal_dead:0,steal_unconditional:0,
    carry_survival:0,progress25:0,progress50:0,progress75:0,home_half:0,capture:0,drop:0,repeat_pickup:0,reverse:0};
  out.approach=Number(Math.abs(s.pos[0]-s.pos[2])+Math.abs(s.pos[1]-s.pos[3])<b.distance);
  out.sugar=Number(b.trap[1]===0&&s.trapped[1]>0);out.kill=Number(info.died[1]);
  const rising=b.carry[0]<0&&s.bunCarried[0]===1;
  const enemyCell=s.centerCell(0);const cage=s._bunBaseTeam(...enemyCell);
  // Native _bunUpdate happens before respawn: old dead/respawn1 is vulnerable.
  const fromCage=rising&&cage===1&&b.stock[1][1]>s.bunStored[1][1];
  out.steal_trap=Number(fromCage&&(s.trapped[1]>0));
  out.steal_dead=Number(fromCage&&(info.died[1]||!b.alive[1]||b.respawn[1]>0));
  out.steal_unconditional=Number(fromCage&&!out.steal_trap&&!out.steal_dead);
  if(rising){out.repeat_pickup=Number(ledger.seen);ledger.seen=true;}
  out.capture=Number(b.carry[0]===1&&s.bunCarried[0]<0&&s.bunStored[0][1]>b.stock[0][1]);
  out.drop=Number(b.carry[0]===1&&info.died[0]);
  out.carry_survival=Number(s.bunCarried[0]===1&&s.alive[0]);
  if(table){
    const cell=s.centerCell(0),prev=b.cell;const index=cell[0]*QQT.W+cell[1],oldidx=prev[0]*QQT.W+prev[1];
    const region=table.regions[0][cell[0]][cell[1]],oldregion=table.regions[0][prev[0]][prev[1]];
    if(b.carry[0]===1||s.bunCarried[0]===1){
      if(ledger.high<0)ledger.high=oldregion;
      for(const [threshold,key] of [[8,'progress25'],[16,'progress50'],[24,'progress75']])
        if(region>ledger.high&&ledger.high<threshold&&region>=threshold)out[key]=1;
      out.home_half=Number(region>ledger.high&&table.home[0][cell[0]][cell[1]]&&!(ledger.marks&8));
      if(out.home_half)ledger.marks|=8;
      out.reverse=Number(region<oldregion);ledger.high=Math.max(ledger.high,region);
    }
    void index;void oldidx;
  }
  return out;
}
async function run(modelPath,out,task,regionsFile){
  assert.ok(tasks.includes(task));assert.equal(process.env.CUDA_VISIBLE_DEVICES||'','');
  const loaded=await inference.loadModel(modelPath);const tables=JSON.parse(fs.readFileSync(regionsFile,'utf8'));
  const rows=[];const start=Date.now();
  for(let game=0;game<16;game++){
    const seed=(2026101131+tasks.indexOf(task)*7919+game*104729)>>>0;
    const s=setup(task,seed);const rng=QQT.mulberry32(seed^0x515154);const hunter=authority.createHunter('normal');
    hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:`je05-${task}-${game}`,seed,ruleset:'bun',max_ticks:300,metadata:{}});
    const counts={deaths:0,self:0,trade:0,bombs:0,kills:0};const funnel={};const ledger=defaultLedger();const trapCause=[[false,false],[false,false]];const tracer=new authority.AttackTracer(s);
    const step=async settling=>{
      const b={carry:s.bunCarried.slice(),stock:s.bunStored.map(x=>x.slice()),alive:s.alive.slice(),trap:s.trapped.slice(),respawn:s.bunRespawn.slice(),cell:s.centerCell(0),
        distance:Math.abs(s.pos[0]-s.pos[2])+Math.abs(s.pos[1]-s.pos[3])};
      const pre={pos:Array.from(s.pos),alive:Array.from(s.alive),fuse:Int16Array.from(s.fuse),blast:s.blastCap[0]};const tick=s.t;
      const raw=await loaded.model.act(s,0,rng);const bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},1,rng);
      const a=[[raw[0],settling?0:Number(raw[1]),settling?0:Number(raw[2]||0)],
        [bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)]];
      if(settling)s.done=false;
      const info=s.step(a);tracer.step(s,tick,pre,info);counts.deaths+=Number(info.died[0]);counts.bombs+=Number(info.placed[0]);
      counts.trade+=Number(info.mutualDeath);
      for(let victim=0;victim<2;victim++){
        if(b.trap[victim]===0&&s.trapped[victim]>0)trapCause[victim]=info.causalDamageSource[victim].map(Boolean);
        if(info.died[victim]){
          const killer=1-victim;
          const pop=b.trap[victim]>0&&s.alive[killer]&&s.trapped[killer]===0&&Math.abs(s.pos[victim*2]-s.pos[killer*2])<41/40&&Math.abs(s.pos[victim*2+1]-s.pos[killer*2+1])<41/40;
          const cause=pop?[victim===1,victim===0]:(b.trap[victim]>0?trapCause[victim]:info.causalDamageSource[victim]);
          if(victim===1&&!info.mutualDeath&&cause[0]&&!cause[1])counts.kills++;
          if(victim===0&&!info.mutualDeath&&cause[0]&&!cause[1])counts.self++;
          trapCause[victim]=[false,false];
        }
      }
      const f=updateFunnel(s,b,info,ledger,tables[task==='carry_home'?1:0]);
      for(const [k,v]of Object.entries(f))funnel[k]=(funnel[k]||0)+v;
    };
    while(!s.done&&s.t<300)await step(false);
    const primary=s.t;s.maxSteps=primary+125;
    while(residual.hasHazards(s)&&s.t<primary+124)await step(true);
    assert.ok(!residual.hasHazards(s));
    funnel.threat=tracer.placements.filter(p=>p.threat).length;funnel.pressure=tracer.placements.filter(p=>p.pressure).length;
    rows.push({game,seed,counts,funnel,attack_trace:tracer.toJSON(),primary_ticks:primary,final_alive:s.alive[0],safe_win:counts.kills>0&&counts.deaths===0&&s.alive[0],hazards_remaining:false});
  }
  const metric=key=>rows.filter(r=>r.funnel[key]>0).length;
  const result={games:16,capture_episodes:metric('capture'),steal_episodes:rows.filter(r=>r.funnel.steal_trap||r.funnel.steal_dead||r.funnel.steal_unconditional).length,
    death_episodes:rows.filter(r=>r.counts.deaths>0).length,self_episodes:rows.filter(r=>r.counts.self>0).length,trade_episodes:rows.filter(r=>r.counts.trade>0).length,
    safe_win:rows.filter(r=>r.safe_win).length,funnel:Object.fromEntries(Object.keys(rows[0].funnel).map(k=>[k,rows.reduce((a,r)=>a+r.funnel[k],0)]))};
  const document={passed:true,task,results:{[task]:result},episodes:rows,wall_seconds:(Date.now()-start)/1000,cpu_only:true,native_trap:true,native_items:true,
    seed_base:2026101131,protocol:'je05_native_transport_web_v2',screening_only:true,model_json_sha256:loaded.sha};
  fs.writeFileSync(out+'.tmp',JSON.stringify(document));fs.renameSync(out+'.tmp',out);
}
if(require.main===module){const [model,out,task,regions]=process.argv.slice(2);run(model,out,task,regions).catch(e=>{console.error(e);process.exitCode=1;});}
module.exports={setup,updateFunnel,run};
