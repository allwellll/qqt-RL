'use strict';
// Private CPU stream: exact production branches; only compact A1 is published.
const fs=require('fs'),assert=require('assert'),readline=require('readline');
const Q=require('../web/sim'),P=require('../web/review_protocol'),G=require('./generate_diagnostic_replays');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json'))[0];
function instrument(s){
  let bun,drops,explosions;
  const update=s._bunUpdate.bind(s),drop=s._bunDrop.bind(s),resolve=s._resolveExplosions.bind(s);
  s._bunUpdate=()=>{bun=P.state(s);return update();};
  s._bunDrop=p=>{if(s.bunCarried[p]>=0)drops.push({player:p,origin:s.bunCarried[p],cell:s.centerCell(p)});return drop(p);};
  s._resolveExplosions=(...args)=>{const result=resolve(...args);explosions=result.sources.map(x=>({...x,covered:Array.from(x.covered),horzCovered:Array.from(x.horzCovered),vertCovered:Array.from(x.vertCovered)}));return result;};
  return row=>{bun=null;drops=[];explosions=[];const info=s.step(P.unpack(row));return {state:P.state(s),info,actions:row,legal:s.legalMask(),bun_before:bun,drops,explosions};};
}
function clone(player){
  const fresh=P.setup(player.document.meta,level);
  for(let i=0;i<player.draws;i++)fresh.rng();
  const data=structuredClone(Object.fromEntries(Object.entries(player.sim).filter(([,v])=>typeof v!=='function')));
  const s=Object.assign(Object.create(Q.Sim.prototype),data);s.rng=fresh.rng;return s;
}
function softmax(logits,mask){const peak=Math.max(...logits.filter((_,i)=>mask[i])),p=logits.map((v,i)=>mask[i]?Math.exp(v-peak):0),total=p.reduce((a,b)=>a+b,0);return p.map(v=>v/total);}
async function main(){
  assert.equal(process.env.CUDA_VISIBLE_DEVICES,'');
  const file=process.argv[2],doc=JSON.parse(fs.readFileSync(file)),player=new P.Player(doc,level),model=await G.load();
  const selectedTicks=process.argv[3]?new Set(process.argv[3].split(',').map(Number)):null;
  const trace=readline.createInterface({input:fs.createReadStream(file.replace(/\.json$/,'.trace.jsonl')),crlfDelay:Infinity})[Symbol.asyncIterator]();
  const first=JSON.parse((await trace.next()).value);assert.deepEqual(first.state,P.state(player.sim));
  process.stdout.write(JSON.stringify({initial:first})+'\n');
  for(let tick=0;tick<doc.actions.length;tick++){
    const expected=JSON.parse((await trace.next()).value);
    if(selectedTicks&&!selectedTicks.has(tick+1)){player.step();continue;}
    const taken=doc.actions[tick],a=await model.act(player.sim,Q.mulberry32(1));assert.deepEqual(a,taken.slice(0,2),'actor re-inference');
    const dec=model.getDecision(),pm=softmax(dec.move,dec.move_mask),pa=softmax(dec.ability,dec.ability_mask),candidates=[];
    for(let move=0;move<5;move++)for(let ability=0;ability<3;ability++){
      if(!dec.move_mask[move]||!dec.ability_mask[ability])continue;
      const s=clone(player),row=[move,ability,taken[2],taken[3]],entry=instrument(s)(row);
      // Snapshot/RNG equality is checked on every taken branch, including drops.
      if(move===taken[0]&&ability===taken[1]){
        assert.deepEqual(entry.state,expected.state,`clone/RNG state @${tick+1}`);
        assert.deepEqual(JSON.parse(JSON.stringify(entry.info)),expected.info,`clone/RNG info @${tick+1}`);
      }
      const obs=s.encodeObsJAX(0,24),state=Float32Array.from(s.encodeStateJAX(0,'bun24-trap-timer-v2'));
      candidates.push({action:[move,ability],probability:pm[move]*pa[ability],entry,
        obs:Buffer.from(obs.buffer,obs.byteOffset,obs.byteLength).toString('base64'),
        global:Buffer.from(state.buffer).toString('base64'),terminal:s.done});
    }
    const packet={tick:tick+1,taken:taken.slice(0,2),candidates};
    if(!process.stdout.write(JSON.stringify(packet)+'\n'))await new Promise(resolve=>process.stdout.once('drain',resolve));
    player.step();
  }
  await model.session.release();
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
module.exports={clone,instrument,softmax};
