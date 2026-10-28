'use strict';
// Continuous native trajectories. Initial trap fixture is explicitly separate
// from natural Hunter/full kill evidence. All evaluation uses real Hunter.
const fs=require('fs'),assert=require('node:assert/strict'),crypto=require('crypto');
const QQT=require('../web/sim.js'),authority=require('./eval_web_hunter.js');
const inference=require('./je05_web_hunter.js'),residual=require('./eval_web_hunter_residual.js');
const previous=require('./je08_web_diagnostics.js'),tasks=require('./je05_web_tasks.js');
const manifest={schema:'je09.continuous-native-bridge/v1',seed:2026101529,holdout_seed:2026101539,
  games:16,assist:false,tasks:['drop_return','near_steal','full','reachable_near'],ticks:[1200,1200,2400,1200],
  pickup_return_window:120,seats:'drop alternating; near/full team0',
  fixture:'drop_return enemy own-carrier trap1, declared causal window; not policy kill evidence',
  denominator:'all 16 episodes; natural eligible-drop episodes separately reported'};
function hash(x){return crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');}
class Tracker extends previous.Tracker {
  constructor(team=0,allowSteal=false){super(team);this.allowSteal=allowSteal;this.toward=0;this.stationary=0;this.reverse=0;
    this.first=null;this.startDistance=null;this.returnTick=null;this.marks=[false,false,false];this.broken=false;
    this.death=false;this.self=false;this.trade=false;this.lost=false;this.trace=[];this.maps=null;this.topology=null;}
  observe(s,b,info){
    const team=this.team,enemy=1-team,oldPickup=this.pickupTick,target=this.target[enemy];
    const topology=Array.from(s.brick).join('');
    if(!this.maps||this.topology!==topology){this.topology=topology;this.maps=[0,1].map(t=>{
      const goals=[],base=s.bunBases[t];for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++)goals.push(r*QQT.W+c);
      return previous.distance(s.wall,s.brick,goals);});this.dropMap=null;}
    const cell=s.centerCell(team),index=cell[0]*QQT.W+cell[1];
    const old=Math.floor(b.pos[team*2])*QQT.W+Math.floor(b.pos[team*2+1]);
    if(target!==null&&b.loose[target*2+enemy]>0&&this.credit[enemy]===team){
      if(!this.dropMap||this.dropTarget!==target){this.dropMap=previous.distance(s.wall,s.brick,[target]);this.dropTarget=target;}
      if(this.dropMap[index]<32767&&this.dropMap[old]<32767){const delta=this.dropMap[old]-this.dropMap[index];
        this.toward+=delta;this.stationary+=Number(delta===0);this.reverse+=Number(delta<0);}
    }
    const ownCause=(b.trap[team]>0?this.trapCause[team]:info.causalDamageSource[team]).slice();
    super.observe(s,b,info);
    const acquired=b.carry[team]<0&&s.bunCarried[team]===enemy;
    if(this.first===null&&((this.pickupTick!==null&&oldPickup===null)||(this.allowSteal&&acquired))){
      this.first=s.t;this.startDistance=this.maps[team][index];this.trace.push({tick:s.t,event:'first_pickup',distance:this.startDistance});}
    if(this.first!==null&&!this.broken){
      const d=this.maps[team][index];
      if(s.bunCarried[team]===enemy&&b.carry[team]===enemy){
        if(d<this.startDistance&&s.t-this.first<=120&&this.returnTick===null){this.returnTick=s.t;this.trace.push({tick:s.t,event:'pickup_return',distance:d});}
        for(const [i,passed] of [d<=.75*this.startDistance,d<=.5*this.startDistance,d<=this.maps[enemy][index]].entries())
          if(passed&&!this.marks[i]){this.marks[i]=true;this.trace.push({tick:s.t,event:['progress25','progress50','home_half'][i],distance:d});}
      }
      if(info.died[team]){this.broken=true;this.lost=true;}
    }
    this.death ||= Boolean(info.died[team]);this.trade ||= Boolean(info.mutualDeath);
    const cause=ownCause;
    this.self ||= Boolean(info.died[team]&&cause[team]&&!cause[enemy]&&!info.mutualDeath);
    if(this.trace.length<32&&(info.died[enemy]||this.pickupTick!==oldPickup))
      this.trace.push({tick:s.t,event:info.died[enemy]?'enemy_death':'pickup',eligible:this.eligibleDrops,carried:s.bunCarried.slice()});
  }
  result(){return {...super.result(),eligible_drop:this.eligibleDrops>0,toward_drop:this.toward>0,
    toward_drop_net:this.toward,arrival:this.arrivalTick!==null,first_pickup:this.first!==null,
    pickup_return:this.returnTick!==null,progress25:this.marks[0],progress50:this.marks[1],home_half:this.marks[2],
    capture:this.first!==null&&!this.broken&&this.captureTick!==null,death:this.death,self:this.self,trade:this.trade,
    stationary_ticks:this.stationary,reverse_ticks:this.reverse,lost:this.lost,trace:this.trace};}
}
function summarize(rows){const names=['eligible_drop','toward_drop','arrival','first_pickup','pickup_return','progress25','progress50','home_half','capture','death','self','trade','lost'];
  return {games:rows.length,...Object.fromEntries(names.map(k=>[k,rows.filter(r=>r[k]).length])),episodes:rows};}
function setup(task,seed,team,table,ticks){
  const s=task==='drop_return'?previous.setup(task,seed,team,table,ticks):tasks.setup(task==='reachable_near'?'near_steal':task,seed);
  if(task==='reachable_near')s.brick.fill(0);
  s.maxSteps=ticks;return s;
}
async function run(model,out,regions,mode='evaluation'){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES||'','');
  const loaded=await inference.loadModel(model),table=JSON.parse(fs.readFileSync(regions,'utf8')),results={},begin=Date.now();
  const seedBase=mode==='holdout'?manifest.holdout_seed:manifest.seed;
  if(fs.existsSync(out)){
    const cached=JSON.parse(fs.readFileSync(out,'utf8'));
    assert.equal(cached.model_json_sha256,loaded.sha);assert.equal(cached.seed_base,seedBase);
    assert.equal(cached.mode,mode);assert.equal(cached.assist,false);
    Object.assign(results,cached.results);
  }
  for(const [taskIndex,task]of manifest.tasks.entries()){
    if(results[task])continue;
    const episodes=[],ticks=mode==='smoke'?8:manifest.ticks[taskIndex],games=mode==='smoke'?2:manifest.games;
    for(let game=0;game<games;game++){
      const seed=(seedBase+taskIndex*7919+game*104729)>>>0,team=task==='drop_return'?game%2:0,enemy=1-team;
      const s=setup(task,seed,team,table[1],ticks);
      const tracker=new Tracker(team,task==='near_steal'||task==='reachable_near'),rng=QQT.mulberry32(seed^0x515154),hunter=authority.createHunter('normal');
      if(task==='drop_return')tracker.trapCause[enemy][team]=true;
      hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'je09-'+mode+'-'+task+'-'+game,seed,ruleset:'bun',max_ticks:ticks,metadata:{}});
      const step=async settling=>{
        const b=previous.snapshot(s),raw=await loaded.model.act(s,team,rng);
        const bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng);
        const actions=[];actions[team]=[raw[0],settling?0:Number(raw[1]),settling?0:Number(raw[2]||0)];
        actions[enemy]=[bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)];
        if(settling)s.done=false;tracker.observe(s,b,s.step(actions));
      };
      while(!s.done&&s.t<ticks)await step(false);
      const primary=s.t;s.maxSteps=primary+125;
      while(residual.hasHazards(s)&&s.t<primary+124)await step(true);
      assert.ok(!residual.hasHazards(s));episodes.push({game,seed,team,...tracker.result(),final_alive:s.alive[team],primary_ticks:primary});
    }
    results[task]=summarize(episodes);
  }
  const data={passed:true,assist:false,smoke:mode==='smoke',mode,seed_base:seedBase,manifest,manifest_sha256:hash(manifest),
    model_json_sha256:loaded.sha,results,wall_seconds:(Date.now()-begin)/1000,screening_only:true};
  fs.writeFileSync(out+'.tmp',JSON.stringify(data));fs.renameSync(out+'.tmp',out);return data;
}
if(require.main===module){const [model,out,regions,mode]=process.argv.slice(2);run(model,out,regions,mode).catch(e=>{console.error(e);process.exitCode=1;});}
module.exports={Tracker,summarize,run,setup,manifest};
