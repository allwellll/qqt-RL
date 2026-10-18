'use strict';
// CPU-only private branch stream. Full states/logits never enter public assets.
const fs=require('fs'),assert=require('node:assert/strict'),crypto=require('crypto');
const P=require('../web/review_protocol'),G=require('./generate_diagnostic_replays');
const {instrument,softmax}=require('./diagnostic_counterfactual');
const hash=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
async function main(){
 assert.equal(process.env.CUDA_VISIBLE_DEVICES,'');
 const doc=JSON.parse(fs.readFileSync(process.argv[2])),level=require('./eval_web_hunter').loadLevel(),p=new P.Player(doc,level),actor=await G.load();
 const clone=()=>{const s=P.replaySetup(doc.meta,level);for(let i=0;i<p.draws;i++)s.rng();const rng=s.rng;Object.assign(s,structuredClone(Object.fromEntries(Object.entries(p.sim).filter(([,v])=>typeof v!=='function'))));s.rng=rng;return s;};
 const write=async x=>{if(!process.stdout.write(JSON.stringify(x)+'\n'))await new Promise(r=>process.stdout.once('drain',r));};
 await write({initial:{state:P.state(p.sim),legal:p.sim.legalMask()}});
 for(let i=0;i<doc.actions.length;i++){
  const row=doc.actions[i],taken=await actor.act(p.sim,()=>.5);assert.deepEqual(taken,row.slice(0,2),'same checkpoint Actor action');
  const dec=actor.getDecision(),pm=softmax(dec.move,dec.move_mask),pa=softmax(dec.ability,dec.ability_mask),before=P.state(p.sim),cs=[];
  for(let m=0;m<5;m++)for(let a=0;a<3;a++)if(dec.move_mask[m]&&dec.ability_mask[a]){
   const s=clone(),entry=instrument(s)([m,a,...row.slice(2)]),obs=s.encodeObsJAX(0,24),gv=Float32Array.from(s.encodeStateJAX(0,'bun24-trap-timer-v2'));
   cs.push({action:[m,a],probability:pm[m]*pa[a],entry,terminal:s.done,obs:Buffer.from(obs.buffer,obs.byteOffset,obs.byteLength).toString('base64'),global:Buffer.from(gv.buffer).toString('base64')});
  }
  p.step();const selected=cs.find(c=>c.action.every((a,k)=>a===taken[k]));assert.deepEqual(selected.entry.state,P.state(p.sim),'taken branch/RNG exact');
  await write({tick:i+1,taken,candidates:cs,decision:{action:taken,move_mask:dec.move_mask,ability_mask:dec.ability_mask,move_probability:pm,ability_probability:pa,legal_count:cs.length},before_hash:hash(before),after_hash:hash(P.state(p.sim))});
 }
 await actor.session.release();
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
