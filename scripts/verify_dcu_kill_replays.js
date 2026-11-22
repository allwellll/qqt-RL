'use strict';
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {create}=require('./kill_mode_bridge');
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
function verify(doc){
 assert.equal(doc.schema,'qqt.kill-checkpoint-replays/v1');
 for(const [name,hash]of Object.entries(doc.source_sha256))assert.equal(digest(fs.readFileSync(name)),hash,name);
 for(const game of doc.replays){
  const s=create(game.spec.seed),hashes=new Map(game.state_sha256),metrics=[[0,0,0,0,0,0],[0,0,0,0,0,0]];
  function check(){if(hashes.has(s.t))assert.equal(digest(JSON.stringify(s.snapshotReplay())),hashes.get(s.t),`tick ${s.t}`);}
  check();assert.equal(game.actions.length,1800);
  for(const row of game.actions){
   const info=s.step([[row[0],+(row[1]===1),+(row[1]===2)],[row[2],+(row[3]===1),+(row[3]===2)]]);
   for(let p=0;p<2;p++)[+info.creditedKill[p],+info.died[p],+info.ownBombDefeat[p],+info.placed[p],+(s.trapped[p]>0),s.done?(s.winner===p?1:s.winner===null?.5:0):0].forEach((v,k)=>metrics[p][k]+=v);
   check();
  }
  assert(s.done);assert.equal(s.winner,game.summary.winner);assert.deepEqual(s.killCount,game.summary.kill_count);assert.deepEqual(metrics,game.summary.metrics);
 }
 return {status:'PASS_INDEPENDENT_REPLAY',games:doc.replays.length,ticks:doc.replays.length*1800};
}
if(require.main===module)console.log(JSON.stringify(verify(JSON.parse(fs.readFileSync(process.argv[2])))));
module.exports={verify};
