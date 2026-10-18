'use strict';
// Assist-OFF native event and topology diagnostics; scores are not promotion.
const fs=require('fs'),assert=require('node:assert/strict'),crypto=require('crypto');
const QQT=require('../web/sim.js'),authority=require('./eval_web_hunter.js');
const inference=require('./je05_web_hunter.js'),residual=require('./eval_web_hunter_residual.js');
const manifest={schema:'je08.topology-native-diagnostic/v1',seed:2026101408,games:16,
  tasks:['drop_return','carry_near','carry_mid','carry_far'],ticks:600,assist:false,
  map:806,seats:'alternating0/1',brick_entry_cost:4,drop_fixture:'enemy own-carrier trap1, predeclared learner-caused diagnostic window; never actor kill evidence'};
const manifestHash=crypto.createHash('sha256').update(JSON.stringify(manifest)).digest('hex');
const INF=32767;
function distance(wall,brick,goals,h=QQT.H,w=QQT.W){
  const d=Array(h*w).fill(INF),visited=Array(h*w).fill(false);
  for(const i of goals)if(!wall[i])d[i]=0;
  for(let step=0;step<h*w;step++){
    let at=-1;for(let i=0;i<h*w;i++)if(!visited[i]&&(at<0||d[i]<d[at]))at=i;
    if(at<0||d[at]>=INF)break;visited[at]=true;
    const r=Math.floor(at/w),c=at%w,cost=1+3*Number(Boolean(brick[at]));
    for(const [dr,dc]of [[-1,0],[1,0],[0,-1],[0,1]]){
      const rr=r+dr,cc=c+dc;if(rr<0||rr>=h||cc<0||cc>=w||wall[rr*w+cc])continue;
      const k=rr*w+cc;d[k]=Math.min(d[k],d[at]+cost);
    }
  }
  return d;
}
function snapshot(s){return {carry:s.bunCarried.slice(),loose:Array.from(s.bunLoose),stock:s.bunStored.map(x=>x.slice()),
  pos:Array.from(s.pos),alive:s.alive.slice(),trap:s.trapped.slice(),brick:Array.from(s.brick)};}
class Tracker {
  constructor(team=0){this.team=team;this.nonce=[0,0];this.target=[null,null];this.credit=[-1,-1];this.trapCause=[[false,false],[false,false]];
    this.dropTick=null;this.pickupTick=null;this.captureTick=null;this.arrivalTick=null;this.net=0;this.positive=0;this.negative=0;this.invalid=0;
    this.eligibleDrops=0;this.ownPickup=0;this.phaseEnds=0;}
  observe(s,b,info){
    const team=this.team,enemy=1-team;
    for(let victim=0;victim<2;victim++){
      if(b.trap[victim]===0&&s.trapped[victim]>0)this.trapCause[victim]=info.causalDamageSource[victim].map(Boolean);
      if(info.died[victim]){
        const [r,c]=s.centerCell(victim),cell=r*QQT.W+c,origin=b.carry[victim];
        const cause=b.trap[victim]>0?this.trapCause[victim]:info.causalDamageSource[victim];
        if(origin>=0&&s.bunLoose[cell*2+origin]>b.loose[cell*2+origin]){
          this.nonce[origin]++;this.target[origin]=cell;
          this.credit[origin]=cause[1-victim]&&!cause[victim]&&s.alive[1-victim]&&!info.mutualDeath?1-victim:-1;
          if(origin===enemy&&this.credit[origin]===team){this.dropTick=s.t;this.eligibleDrops++;}
        }
        this.trapCause[victim]=[false,false];
      }
    }
    const acquired=b.carry[team]<0&&s.bunCarried[team]>=0,origin=s.bunCarried[team];
    const target=this.target[enemy],at=s.centerCell(team),index=at[0]*QQT.W+at[1];
    if(target!==null&&this.credit[enemy]===team&&b.loose[target*2+enemy]>0){
      if(index===target&&this.arrivalTick===null)this.arrivalTick=s.t;
      if(acquired&&origin===enemy&&index===target&&s.bunLoose[target*2+enemy]<b.loose[target*2+enemy]&&this.pickupTick===null)this.pickupTick=s.t;
    }
    if(acquired&&origin===team)this.ownPickup++;
    if(acquired&&origin===enemy&&this.pickupTick===null)this.invalid++;
    const captured=b.carry[team]===enemy&&s.bunStored[team][enemy]>b.stock[team][enemy];
    if(captured&&this.captureTick===null)this.captureTick=s.t;
    if(b.carry[team]===enemy&&s.bunCarried[team]!==enemy)this.phaseEnds++;
    const changed=b.brick.some((v,i)=>Boolean(v)!==Boolean(s.brick[i]));
    if(b.carry[team]===enemy&&s.bunCarried[team]===enemy&&s.alive[team]&&!info.died[team]&&!info.mutualDeath&&!changed){
      const base=s.bunBases[team],goals=[];
      for(let r=base[0];r<base[0]+3;r++)for(let c=base[1];c<base[1]+3;c++)goals.push(r*QQT.W+c);
      const map=distance(s.wall,s.brick,goals),old=Math.floor(b.pos[team*2])*QQT.W+Math.floor(b.pos[team*2+1]);
      if(map[old]<INF&&map[index]<INF){const delta=map[old]-map[index];this.net+=delta;this.positive+=Math.max(0,delta);this.negative+=Math.min(0,delta);}
    }
    if(target!==null&&s.bunLoose[target*2+enemy]===0)this.credit[enemy]=-1;
  }
  result(){return {eligible_drop_events:this.eligibleDrops,first_pickup:this.pickupTick!==null,
    arrival_ticks:this.arrivalTick===null||this.dropTick===null?null:this.arrivalTick-this.dropTick,
    pickup_ticks:this.pickupTick===null||this.dropTick===null?null:this.pickupTick-this.dropTick,
    capture:this.captureTick!==null,capture_tick:this.captureTick,carry_net_topology:this.net,
    carry_positive_topology:this.positive,carry_negative_topology:this.negative,
    invalid_enemy_pickup:this.invalid,own_pickup:this.ownPickup,phase_ends:this.phaseEnds};}
}
function setup(task,seed,team,table,ticks=600){
  const s=new QQT.Sim(seed);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
  s.maxSteps=ticks;s.initialHp=1;s.hp=[1,1];s.invuln=[0,0];s.brick.fill(0);
  const enemy=1-team;
  if(task==='drop_return'){
    s.pos[team*2]=5.5;s.pos[team*2+1]=team===0?5.5:9.5;
    s.pos[enemy*2]=5.5;s.pos[enemy*2+1]=team===0?9.5:5.5;
    s.bunCarried[enemy]=enemy;s.bunStored[enemy][enemy]=0;s.trapped[enemy]=1;
  }else{
    const bounds={carry_near:[24,32],carry_mid:[8,24],carry_far:[0,8]}[task],cells=[];
    assert.ok(bounds);
    for(let r=0;r<QQT.H;r++)for(let c=0;c<QQT.W;c++)
      if(!s.wall[r*QQT.W+c]&&table.regions[team][r][c]>=bounds[0]&&table.regions[team][r][c]<bounds[1]&&s._bunBaseTeam(r,c)!==team)cells.push([r,c]);
    const [r,c]=cells[seed%cells.length];s.pos[team*2]=r+.5;s.pos[team*2+1]=c+.5;
    s.bunCarried[team]=enemy;s.bunStored[enemy][enemy]=0;
  }
  s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];s._gen++;return s;
}
async function run(model,out,regions,smoke=false){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES||'','');
  const loaded=await inference.loadModel(model),tables=JSON.parse(fs.readFileSync(regions,'utf8')),results={},start=Date.now();
  for(const task of manifest.tasks){
    const episodes=[];
    for(let game=0;game<(smoke?2:16);game++){
      const seed=(manifest.seed+manifest.tasks.indexOf(task)*7919+game*104729)>>>0,team=game%2,enemy=1-team;
      const s=setup(task,seed,team,tables[1],smoke?8:600),tracker=new Tracker(team),rng=QQT.mulberry32(seed^0x515154),hunter=authority.createHunter('normal');
      if(task==='drop_return')tracker.trapCause[enemy][team]=true;
      hunter.adapter.reset({schema:'qqt.bot.context/v1',episode_id:'je08-'+task+'-'+game,seed,ruleset:'bun',max_ticks:600,metadata:{}});
      const step=async settling=>{
        const b=snapshot(s),raw=await loaded.model.act(s,team,rng);
        const bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng);
        const actions=[];actions[team]=[raw[0],settling?0:Number(raw[1]),settling?0:Number(raw[2]||0)];
        actions[enemy]=[bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)];
        if(settling)s.done=false;const info=s.step(actions);tracker.observe(s,b,info);
      };
      while(!s.done&&s.t<(smoke?8:600))await step(false);
      const primary=s.t;s.maxSteps=primary+125;
      while(residual.hasHazards(s)&&s.t<primary+124)await step(true);
      assert.ok(!residual.hasHazards(s));episodes.push({game,seed,team,...tracker.result(),final_alive:s.alive[team]});
    }
    results[task]={games:episodes.length,first_pickups:episodes.filter(x=>x.first_pickup).length,captures:episodes.filter(x=>x.capture).length,
      net_topology:episodes.reduce((sum,x)=>sum+x.carry_net_topology,0),episodes};
  }
  const d={passed:true,assist:false,protocol:manifest,manifest_sha256:manifestHash,model_json_sha256:loaded.sha,smoke,
    results,screening_only:true,promotion_evidence:false,wall_seconds:(Date.now()-start)/1000};
  fs.writeFileSync(out+'.tmp',JSON.stringify(d));fs.renameSync(out+'.tmp',out);return d;
}
if(require.main===module){const [model,out,regions,smoke]=process.argv.slice(2);run(model,out,regions,smoke==='smoke').catch(e=>{console.error(e);process.exitCode=1;});}
module.exports={distance,Tracker,snapshot,setup,manifest,manifestHash,run};
