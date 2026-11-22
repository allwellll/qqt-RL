'use strict';
// CPU evaluation only. Record complete fresh games without training or resets.
const readline=require('node:readline'),crypto=require('node:crypto');
const {World,reward}=require('./kill_mode_bridge');
const hash=sim=>crypto.createHash('sha256').update(JSON.stringify(sim.snapshotReplay())).digest('hex');
class RecordingWorld extends World {
  constructor(specs){
    super(specs.map(s=>s.seed));this.specs=specs;
    this.records=specs.map((spec,i)=>({spec,actions:[],events:[],state_sha256:[[0,hash(this.sims[i])]],
      metrics:[[0,0,0,0,0,0],[0,0,0,0,0,0]]}));
  }
  step(actions){
    if(actions.length!==this.sims.length)throw Error('Invalid action batch');
    const rewards=[],done=[],metrics=[];
    this.sims.forEach((sim,i)=>{
      if(sim.done)throw Error('Attempt to advance completed game');
      const pair=actions[i];
      if(pair.length!==2||pair.some(a=>a.length!==2||!Number.isInteger(a[0])||a[0]<0||a[0]>4||!Number.isInteger(a[1])||a[1]<0||a[1]>2))throw Error('Invalid action');
      const before={trapped:sim.trapped.slice()},alive=sim.alive.slice();
      const info=sim.step(pair.map(([m,a])=>[m,Number(a===1),Number(a===2)])),record=this.records[i];
      record.actions.push(pair.flat());
      const counts=[0,1].map(p=>[+info.creditedKill[p],+info.died[p],+info.ownBombDefeat[p],+info.placed[p],+(sim.trapped[p]>0),sim.done?(sim.winner===p?1:sim.winner===null?.5:0):0]);
      for(let p=0;p<2;p++){
        counts[p].forEach((v,k)=>record.metrics[p][k]+=v);
        for(const [name,yes]of [['kill',info.creditedKill[p]],['death',info.died[p]],['self_death',info.ownBombDefeat[p]],['respawn',!alive[p]&&sim.alive[p]]])if(yes)record.events.push([sim.t,p,name]);
      }
      if(sim.t%180===0||sim.done)record.state_sha256.push([sim.t,hash(sim)]);
      if(sim.done)record.summary={ticks:sim.t,full_game:sim.t===1800,winner:sim.winner,kill_count:sim.killCount.slice(),metrics:record.metrics};
      rewards.push(reward(before,sim,info));done.push(+sim.done);metrics.push(counts);
    });
    return this.observe({rewards,done,metrics});
  }
}
if(require.main===module){
 let world;
 const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
 rl.on('line',line=>{try{
   const q=JSON.parse(line);let response;
   if(q.command==='init'){world=new RecordingWorld(q.specs);response=world.observe();}
   else if(q.command==='step')response=world.step(q.actions);
   else if(q.command==='finish'){
     if(!world.sims.every(s=>s.done))throw Error('Only complete games may be exported');
     response=world.observe({replays:world.records});
   }else throw Error('Unknown command');
   process.stdout.write(JSON.stringify(response)+'\n');
 }catch(e){process.stdout.write(JSON.stringify({error:e.stack})+'\n');process.exitCode=1;rl.close();}});
}
module.exports={RecordingWorld,hash};
