'use strict';
const fs=require('fs'),assert=require('assert'),P=require('../web/review_protocol'),A=require('./eval_web_hunter'),M=require('./bomb_threat_p0_jitter_metrics');
const B='runs/bomb_threat_v1/jitter_followup_u800',cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json'));
const read=p=>JSON.parse(fs.readFileSync(p)),D=[[-1,0],[1,0],[0,-1],[0,1]],neigh=i=>D.map(([y,x])=>[Math.floor(i/15)+y,i%15+x]).filter(([y,x])=>y>=0&&y<13&&x>=0&&x<15).map(([y,x])=>y*15+x);
function cast(s,start){const cells=[start],r=Math.floor(start/15),c=start%15;for(const [dy,dx] of D)for(let k=1;k<=s.bombBlast[start];k++){const y=r+dy*k,x=c+dx*k,i=y*15+x;if(y<0||y>=13||x<0||x>=15||s.wall[i])break;cells.push(i);if(s.brick[i]||s.fuse[i]>0)break;}return cells;}
function projected(s,owner){const cover=new Set(),queue=[];for(let i=0;i<195;i++)if(s.fuse[i]>0&&(owner===null||s.owner[i]===owner))queue.push(i);const seen=new Set();for(let k=0;k<queue.length;k++){const i=queue[k];if(seen.has(i))continue;seen.add(i);for(const j of cast(s,i)){cover.add(j);if(j!==i&&s.fuse[j]>0)queue.push(j);}}return cover;}
function objects(s){return Array.from({length:195},(_,i)=>s.wall[i]?'wall':s.pushable[i]?'push_box':s.brick[i]?'brick':s.fuse[i]>0?'live_bomb':null);}
const results=[],danger=[],hunter=[];
const oldcfg=read(cfg.jitter_source+'/PROTOCOL.json');
for(const g of oldcfg.games){
 const source=`${cfg.source}/game_${String(g.index).padStart(3,'0')}`,doc=read(source+'/review.json'),jr=read(`${cfg.jitter_source}/game_${String(g.index).padStart(3,'0')}.json`),p=new P.Player(doc,A.loadLevel());
 const oscillation=new Set();for(const w of jr.windows)if(w.raw_oscillation)for(let i=w.start;i<w.end;i++)oscillation.add(i);
 const detailed=cfg.reversibility_indices.includes(g.index),history=[],changes=[],pushes=[],dangerCounts={},relative={},botMoves=[],botActive=[];let previousObjects=objects(p.sim),cellSeen=new Map(),pickup=[];
 const startPos=Array.from(p.sim.pos.slice(0,2)),home=p.sim.bunBases[0],targets=[];for(let y=home[0];y<home[0]+3;y++)for(let x=home[1];x<home[1]+3;x++)targets.push(y*15+x);
 const sc=p.sim.centerCell(0),birthHome=M.homeDistance(p.sim,sc[0]*15+sc[1],targets);
 const birthSnapshot={wall:Array.from(p.sim.wall),brick:Array.from(p.sim.brick),fuse:Array.from(p.sim.fuse),pushable:Array.from(p.sim.pushable),own_targets:targets,start_cell:sc[0]*15+sc[1]};
 for(let t=0;t<doc.actions.length;t++){
  const s=p.sim,pos=Array.from(s.pos.slice(0,2)),ec=s.centerCell(1),cc=s.centerCell(0),cell=cc[0]*15+cc[1],controlled=!!s.alive[0]&&!s.trapped[0],carry=s.bunCarried[0],beforeObj=objects(s),oldBoxes=s.pushBoxes.map(b=>({o:b.o,cells:Array.from(b.cells),dead:b.dead}));
  botMoves.push(doc.actions[t][2]);botActive.push(!!s.alive[1]&&!s.trapped[1]);
  if(oscillation.has(t)){
   const e=projected(s,1),all=projected(s,null),inside=e.has(cell),edge=inside&&neigh(cell).some(i=>!e.has(i)),outside=!inside&&neigh(cell).some(i=>e.has(i)),linger=s.blastLinger[cell]>0;
   const type=edge?'enemy_blast_edge':inside?'enemy_blast_interior':outside?'adjacent_outside_enemy_blast':linger||all.has(cell)?'other_or_active_flame_risk':'no_observed_blast_danger';dangerCounts[type]=(dangerCounts[type]||0)+1;
   const dy=s.pos[2]-pos[0],dx=s.pos[3]-pos[1],key=`${Math.sign(dy)},${Math.sign(dx)}`;relative[key]=(relative[key]||0)+1;
  }
  if(detailed){const homeDist=M.homeDistance(s,cell,targets),row={tick:t,pos,cell,controlled,carry,home_static:homeDist,home_dynamic:M.homeDistance(s,cell,targets,false,true),objects:beforeObj,move:doc.actions[t][0]};history.push(row);cellSeen.set(cell,t);}
  assert(p.step());
  if(detailed){
   for(let b=0;b<s.pushBoxes.length;b++){const old=oldBoxes[b],box=s.pushBoxes[b];if(old&&JSON.stringify(old.cells)!==JSON.stringify(box.cells))pushes.push({tick_after:s.t,box:b,from:old.cells,to:Array.from(box.cells),actions:doc.actions[t],before_positions:[...pos,...(jr.rows[t].enemy_pos)],attribution:'pending_actual_push_hook'});}
   const afterObj=objects(s);for(let i=0;i<195;i++)if(beforeObj[i]!==afterObj[i])changes.push({tick_after:s.t,cell:i,before:beforeObj[i],after:afterObj[i],bomb_owner:s.owner[i],last_learner_visited_before:cellSeen.get(i)??null});
   if(carry<0&&s.bunCarried[0]===1){const end=t+1,start=Math.max(0,end-300),path=history.slice(start,end).map(r=>({tick:r.tick,pos:r.pos,cell:r.cell}));const ac=s.centerCell(0);path.push({tick:end,pos:Array.from(s.pos.slice(0,2)),cell:ac[0]*15+ac[1]});
    const pathcells=[...new Set(path.map(r=>r.cell))],blocked=pathcells.filter(i=>afterObj[i]),origin=path[0].cell,backToArrival=M.movementField(s,ac[0]*15+ac[1],false,true)[origin];
    pickup.push({pickup_tick:end,path_start:start,origin_before:carry,origin_after:s.bunCarried[0],stock_after:s.bunStored.map(a=>a.slice()),spawn_pos:startPos,birth_home_static:birthHome,arrival_home_static_start:history[start].home_static,
     arrival_path:path,arrival_cells:pathcells,blocked_arrival_cells_at_pickup:blocked.map(i=>({cell:i,kind:afterObj[i],last_traversed:path.filter(r=>r.cell===i).at(-1).tick,changes:changes.filter(c=>c.cell===i)})),
     reverse_arrival_graph_distance:backToArrival,home_static_at_pickup:M.homeDistance(s,ac[0]*15+ac[1],targets),home_dynamic_at_pickup:M.homeDistance(s,ac[0]*15+ac[1],targets,false,true),
     caveat:'reverse300tick path origin may be enemy-side birth/position, not own cage; connectivity is not certified safety',snapshots:[]});
   }
   for(const event of pickup){if(s.t<event.pickup_tick||carry!==1&&s.bunCarried[0]!==1)continue;if(s.t===event.pickup_tick||s.t%200===0||s.done){const ac=s.centerCell(0);event.snapshots.push({tick:s.t,pos:Array.from(s.pos.slice(0,2)),carry:s.bunCarried[0],reverse_arrival_graph_distance:M.movementField(s,ac[0]*15+ac[1],false,true)[event.arrival_path[0].cell],home_static:M.homeDistance(s,ac[0]*15+ac[1],targets),home_dynamic:M.homeDistance(s,ac[0]*15+ac[1],targets,false,true),blocked_arrival_cells:event.arrival_cells.filter(i=>afterObj[i]).map(i=>({cell:i,kind:afterObj[i],changes:changes.filter(c=>c.cell===i&&c.tick_after>=event.path_start)}))});}}
  }
 }
 const rev=M.reversals(botMoves,botActive);hunter.push({index:g.index,group:g.group,move_ticks:botMoves.filter((m,i)=>botActive[i]&&m<4).length,reversals:rev.events.length,intervals:rev.intervals});danger.push({index:g.index,group:g.group,oscillation_ticks:oscillation.size,type_ticks:dangerCounts,enemy_relative_sign_ticks:relative});
 if(detailed){const data={status:'REPLAYED',index:g.index,birth_pos:startPos,birth_home_static:birthHome,birth_snapshot:birthSnapshot,pickups:pickup,object_changes:changes,push_events:pushes,history};fs.writeFileSync(`${B}/reversibility_${g.index}.json`,JSON.stringify(data));results.push({index:g.index,birth_pos:startPos,birth_home_static:birthHome,pickups:pickup,push_events:pushes});}
}
fs.writeFileSync(B+'/REVERSIBILITY.json',JSON.stringify({status:'REPLAYED',games:results}));fs.writeFileSync(B+'/DANGER_AND_HUNTER.json',JSON.stringify({status:'RECOMPUTED',danger,hunter,definitions:cfg.danger}));console.log(JSON.stringify({games:results.length,oscillation_games:danger.length}));
