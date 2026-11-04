'use strict';
// Recompute native trap attribution from saved actions; no policy inference.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto'),P=require('../web/five-item-slots-review-v1'),{ControlFunnel}=require('./je10_control_funnel');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
async function audit(file){
 const doc=JSON.parse(fs.readFileSync(file)),player=new P.Player(doc,level);await player.verify();const p=doc.meta.learner_seat,q=1-p,causal=new ControlFunnel(p);
 const r={bombs:0,enemy_bombs:0,deaths:0,self_deaths:0,deliveries:0,carrying_deaths:0,uses:[0,0,0,0,0]};
 while(player.sim.t<doc.actions.length){const s=player.sim,before={trapped:Array.from(s.trapped)},carried=s.bunCarried[p],stored=s.bunStored[p].slice(),a=doc.actions[s.t];player.step();const i=player.info;causal.observe(before,s,i);
  r.bombs+=Number(i.placed[p]);r.enemy_bombs+=Number(i.placed[q]);r.deaths+=Number(i.died[p]);r.self_deaths+=Number(i.ownBombDefeat[p]);
  if(carried>=0){r.deliveries+=Math.max(0,s.bunStored[p][carried]-stored[carried]);r.carrying_deaths+=Number(i.died[p]);}
  if(a[p*2+1]>=2)r.uses[a[p*2+1]-2]++;
 }
 assert.equal(player.sim.winner,doc.summary.winner);return {...r,causal_kills:causal.metrics.causal_kill,surviving_kills:causal.metrics.surviving_kill,metrics_version:'native-trap-control-funnel-v1',replay_sha256:sha(fs.readFileSync(file))};
}
async function main(){
 const base=process.argv[2],summaries=[];
 for(const model of fs.readdirSync(base))for(const tier of fs.readdirSync(path.join(base,model))){const dir=path.join(base,model,tier);if(!fs.statSync(dir).isDirectory())continue;
  const results=[];for(const f of fs.readdirSync(dir).filter(x=>x.endsWith('.json.result'))){const file=path.join(dir,f),r=JSON.parse(fs.readFileSync(file)),actual=await audit(file.slice(0,-7));for(const k of ['bombs','deaths','deliveries','carrying_deaths','replay_sha256'])assert.deepEqual(r[k],actual[k]);Object.assign(r,actual);fs.writeFileSync(file,JSON.stringify(r));results.push(r);}
  const complete=path.join(dir,'COMPLETE.json');if(fs.existsSync(complete)){const d=JSON.parse(fs.readFileSync(complete));assert.equal(d.results.length,results.length);d.results=results;fs.writeFileSync(complete,JSON.stringify(d));}summaries.push({model,tier,games:results.length});
 }
 fs.writeFileSync(path.join(path.dirname(base),path.basename(base)+'_AUDIT.json'),JSON.stringify({status:'PASS',summaries,scope:'all saved actions rechecked; causal kills track initial native control through death; native ownBombDefeat is not all indirect self-harm'},null,2));console.log('Native replay metric audit PASS',summaries.reduce((n,x)=>n+x.games,0));
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={audit};
