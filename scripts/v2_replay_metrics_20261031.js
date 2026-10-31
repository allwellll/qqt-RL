'use strict';
// Recompute passive diagnostics from verified Review actions, without inference.
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const Q=require('../web/sim-bun-native-v2'),P=require('../../cx_publish_r1_1805_20261030/web/review_protocol');
const base=process.argv[2],cfg=JSON.parse(fs.readFileSync(base+'/MANIFEST.json'));
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
 const rows=[];
 for(const index of [...new Set(cfg.games.map(g=>g.index))]){
  const file=base+'/game_'+String(index).padStart(3,'0')+'/review.json';
  const doc=JSON.parse(fs.readFileSync(file)),entry=cfg.games.find(g=>g.index===index),seat=entry.seat;
  const player=new P.Player(doc,level);await player.verify();
  const audit=new P.Player(doc,level);let connected=0,different=0,placed=0,threatened=0,close=0,eligible=0;
  const bombs=new Map(),resolved=[];let immediateResolved=0;
  for(const action of doc.actions){
   const s=audit.sim,old=[];
   for(let c=0;c<195;c++)if(s.fuse[c]>0&&s.owner[c]===seat){const m=new Uint8Array(195);m[c]=1;old.push({cell:c,due:s.t+s.fuse[c],ray:s._rays(m,new Int16Array(s.bombBlast),s.brick).covered});}
   // Review policy uses the same implicit placement cell as production actions.
   const startCell=s.centerCell(seat),placementCell=startCell[0]*15+startCell[1],newDue=s.t+1+Q.CFG.fuse;
   assert(audit.step());
   const info=audit.info;
   if(info.placed[seat]){
    // Placement precedes movement/explosion. A freshly placed bomb may chain
    // explode in this same step, or replace a fuse=1 bomb. Post-step fuse
    // differencing cannot locate either case; centerCell is production's
    // physical placement cell for Review's four-component actions.
    const at=placementCell,links=old.filter(b=>b.cell!==at&&b.ray[at]);placed++;
    if(links.length)connected++;
    if(links.some(b=>Math.abs(b.due-newDue)>1))different++;
    if(bombs.has(at)){resolved.push(bombs.get(at));bombs.delete(at);}
    if(s.fuse[at]===0)immediateResolved++;
    bombs.set(at,{cell:at,tick:s.t,threat:false});
   }
   const enemy=1-seat;
   const can=s.alive[seat]&&s.trapped[seat]===0&&s.alive[enemy];
   eligible+=Number(can);close+=Number(can&&Math.abs(s.pos[0]-s.pos[2])+Math.abs(s.pos[1]-s.pos[3])<=3);
   for(const [at,b] of bombs){
    if(s.fuse[at]>0){const m=new Uint8Array(195);m[at]=1;const ray=s._rays(m,new Int16Array(s.bombBlast),s.brick).covered;
     const e=Math.floor(s.pos[2*enemy])*15+Math.floor(s.pos[2*enemy+1]);b.threat ||= Boolean(s.alive[enemy]&&ray[e]);
    }else{resolved.push(b);bombs.delete(at);}
   }
  }
  const s=audit.sim;
  assert.equal(placed,entry.expected.placed[seat],'all real placements counted including same-step explosions');
  assert.equal(hash(JSON.stringify(P.state(s))),doc.checks.at(-1)[1]);
  rows.push({index,seed:entry.seed,seat,ticks:s.t,done:s.done,review_sha256:hash(fs.readFileSync(file)),
   placed,immediate_same_step_resolved:immediateResolved,connection_to_old_bombs:connected,distinct_due_connections:different,
   effective_threat_resolved:resolved.filter(b=>b.threat).length,resolved_bombs:resolved.length,
   unknown_terminal_bombs:bombs.size,near_enemy3_ticks:close,near_enemy3_denominator:eligible,
   attribute_final:[s.bombsCap[seat],s.blastCap[seat],s.spdG[seat]]});
 }
 fs.writeFileSync(base+'/R1_REPLAY_METRICS.json',JSON.stringify({status:'RECOMPUTED_FROM_VERIFIED_REVIEW',rows,
  definitions:{connection:'real new bomb is covered by an already live own bomb ray at placement',
   distinct_due:'connected old/new expected fuse expiry differs by more than one tick; not claimed causal combo',
   threat:'live own bomb ray covered live enemy cell on an observed post-step tick; explosion-only instant coverage may be missed',
   near:'own alive/uncontrolled and enemy alive, continuous Manhattan distance <=3',
   attributes:'actual terminal bomb capacity/blast/speed; not training high-water reward'},
  missing:['instant-only threat coverage of same-step exploded new bombs','training reward provenance','causal net tactical value of bomb connection']},null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
