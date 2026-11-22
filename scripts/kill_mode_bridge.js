'use strict';
// Persistent CPU simulator: use the actual live web engine and its adapters.
const readline = require('node:readline');
const Q = require('../web/sim-bun-native-v4-facing');
const opening = require('../web/kill_opening_items');
const respawn = require('../web/self_respawn_preview');
const inputs = require('../web/r3_model_inputs');
const levels = require('../web/assets/maps/levels.json');
inputs.installSim(Q);
const level = levels.find(x => x.qqt_id === 806);
const encoder = Q.TransformerModel.prototype;

function create(seed) {
  const sim = new Q.Sim(seed >>> 0);
  sim.reset(level, {nativeItems:true, nativeTrap:true, killMode:true, teams:[0,1], bananaSlideSpeedPx:480});
  respawn.attach(sim);
  const report = opening.apply(sim);
  if (report.actual !== 24) throw Error('Expected 24 opening growth pickups');
  sim.maxSteps = 1800;
  return sim;
}

// No bubble-count, pickup-count or proximity reward. Only factual combat events.
function reward(before, sim, info) {
  return [0,1].map(p => {
    const enemy=1-p;
    const trap=before.trapped[enemy]===0 && sim.trapped[enemy]>0 &&
      info.physicalDamageSource[enemy][p] && !info.physicalDamageSource[enemy][enemy];
    const trapped=before.trapped[p]===0 && sim.trapped[p]>0;
    const result=5*Number(info.creditedKill[p])-5*Number(info.died[p])-
      2*Number(info.ownBombDefeat[p])+.25*Number(trap)-.25*Number(trapped);
    return result + (sim.done && sim.winner!==null ? (sim.winner===p ? 2 : -2) : 0);
  });
}

class World {
  constructor(seeds) {
    if (!Array.isArray(seeds) || !seeds.length || seeds.some(s=>!Number.isInteger(s))) throw Error('Explicit seeds required');
    this.seeds=seeds.slice(); this.episodes=seeds.map(()=>0); this.sims=seeds.map(create);
  }
  reset(i) {
    this.episodes[i]++;
    this.sims[i]=create((this.seeds[i]+Math.imul(this.episodes[i],0x9e3779b9))>>>0);
  }
  step(actions) {
    if (!Array.isArray(actions)||actions.length!==this.sims.length) throw Error('Action batch length');
    const rewards=[],done=[],metrics=[];
    for(let i=0;i<this.sims.length;i++) {
      const sim=this.sims[i], pair=actions[i];
      if (!Array.isArray(pair)||pair.length!==2||pair.some(a=>a.length!==2||!Number.isInteger(a[0])||!Number.isInteger(a[1])||a[0]<0||a[0]>4||a[1]<0||a[1]>2)) throw Error('Invalid move/ability');
      const before={trapped:sim.trapped.slice()};
      const info=sim.step(pair.map(([move,a])=>[move,Number(a===1),Number(a===2)]));
      rewards.push(reward(before,sim,info)); done.push(Number(sim.done));
      metrics.push([0,1].map(p=>[Number(info.creditedKill[p]),Number(info.died[p]),
        Number(info.ownBombDefeat[p]),Number(info.placed[p]),Number(sim.trapped[p]>0),
        sim.done ? (sim.winner===p ? 1 : sim.winner===null ? .5 : 0) : 0]));
      if (sim.done) this.reset(i);
    }
    return this.observe({rewards,done,metrics});
  }
  observe(extra={}) {
    const n=this.sims.length,obs=new Float32Array(n*2*25*195),state=new Float32Array(n*2*38);
    const mm=[],bm=[],live=[];
    for(let i=0;i<n;i++) {
      const sim=this.sims[i], masks=sim.legalMask();
      mm[i]=[];bm[i]=[];live[i]=[];
      for(let p=0;p<2;p++) {
        const x=inputs.inputs(sim,p),index=i*2+p;
        obs.set(x.obs,index*25*195);state.set(x.state,index*38);
        // Keep the inherited mode row zero, matching exported web actors.
        const active=sim.alive[p] && sim.trapped[p]===0;
        mm[i][p]=active ? masks.mm[p] : [0,0,0,0,1];
        bm[i][p]=active ? encoder._abilityMask.call({abilityActions:3},sim,masks,p) : [1,0,0];
        live[i][p]=Number(active);
      }
    }
    const b64=a=>Buffer.from(a.buffer,a.byteOffset,a.byteLength).toString('base64');
    return {n,obs:b64(obs),state:b64(state),mm,bm,live,...extra};
  }
}

if(require.main===module) {
  let world;
  const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  rl.on('line',line=>{
    try {
      const request=JSON.parse(line);
      if(request.command==='init') {world=new World(request.seeds);process.stdout.write(JSON.stringify(world.observe())+'\n');}
      else if(request.command==='step'&&world) process.stdout.write(JSON.stringify(world.step(request.actions))+'\n');
      else throw Error('Unknown command or uninitialized world');
    } catch(error) {process.stdout.write(JSON.stringify({error:error.message})+'\n');process.exitCode=1;rl.close();}
  });
}
module.exports={World,create,reward};
