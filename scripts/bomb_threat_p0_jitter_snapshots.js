'use strict';
// Exact original states, no model or new rollout. Expose graph blockers explicitly.
const fs=require('fs'),P=require('../web/review_protocol'),A=require('./eval_web_hunter');
const B='runs/bomb_threat_v1/p0_jitter_u800',cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json'));
const out=[];
for(const g of cfg.games){
 const d=JSON.parse(fs.readFileSync(`${B}/game_${String(g.index).padStart(3,'0')}.json`)),rows=d.rows;
 const times=new Set();for(let i=0;i<rows.length;i++)if(rows[i].carry&&(i===0||!rows[i-1].carry))times.add(i);
 if(rows.at(-1).carry)times.add(rows.length-1);if(!times.size)continue;
 const doc=JSON.parse(fs.readFileSync(`${cfg.source}/game_${String(g.index).padStart(3,'0')}/review.json`)),p=new P.Player(doc,A.loadLevel());
 for(const tick of [...times].sort((a,b)=>a-b)){
  p.seek(tick);const s=p.sim,[r,c]=s.centerCell(0),start=r*15+c,anchor=s.bunBases[s.team[0]],targets=[];
  for(let y=anchor[0];y<anchor[0]+3;y++)for(let x=anchor[1];x<anchor[1]+3;x++)targets.push(y*15+x);
  const q=[start],prev=new Int16Array(195).fill(-1);prev[start]=start;let target=null;
  for(let k=0;k<q.length;k++){
   const i=q[k];if(targets.includes(i)){target=i;break;}const y=Math.floor(i/15),x=i%15;
   for(const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]){const yy=y+dy,xx=x+dx,j=yy*15+xx;if(yy<0||yy>=13||xx<0||xx>=15||prev[j]>=0||s.wall[j]||s.pushable[j])continue;prev[j]=i;q.push(j);}
  }
  const path=[];if(target!==null){for(let i=target;;i=prev[i]){path.push(i);if(i===start)break;}path.reverse();}
  out.push({index:g.index,tick,own_cage_anchor:anchor,home_targets:targets,position:s.pos.slice(0,2),static_home:rows[tick].home_static,dynamic_home:rows[tick].home_dynamic,
   witness_path_ignoring_bricks_bombs:path,brick_cells_on_witness:path.filter(i=>s.brick[i]),live_bomb_cells_on_witness:path.filter(i=>s.fuse[i]>0),
   caveat:'shortest length witness ignoring brick and bomb occupancy, not minimum brick count, not executable safe return plan',
   wall:Array.from(s.wall),brick:Array.from(s.brick),fuse:Array.from(s.fuse),pushable:Array.from(s.pushable),crate:Array.from(s.crate)});
 }
}
fs.writeFileSync(B+'/BARRIER_SNAPSHOTS.json',JSON.stringify({status:'RECOMPUTED',snapshots:out}));console.log(JSON.stringify({snapshots:out.length}));
