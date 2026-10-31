'use strict';
// Correct geometry measurements only, preserving all previous inference and actions.
const fs=require('fs'),P=require('../web/review_protocol'),A=require('./eval_web_hunter'),M=require('./bomb_threat_p0_jitter_metrics');
const B='runs/bomb_threat_v1/p0_jitter_u800',cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json'));
for(const g of cfg.games){
 const path=`${B}/game_${String(g.index).padStart(3,'0')}.json`,d=JSON.parse(fs.readFileSync(path));if(d.geometry_version===2)continue;
 const doc=JSON.parse(fs.readFileSync(`${cfg.source}/game_${String(g.index).padStart(3,'0')}/review.json`)),p=new P.Player(doc,A.loadLevel());
 for(const r of d.rows){
  const s=p.sim,c=s.centerCell(0),i=c[0]*15+c[1],own=s.bunBases[s.team[0]],targets=[];
  for(let y=own[0];y<own[0]+3;y++)for(let x=own[1];x<own[1]+3;x++)targets.push(y*15+x);
  r.home_static=M.homeDistance(s,i,targets);r.home_dynamic=M.homeDistance(s,i,targets,false,true);r.home_ignore_bricks_live_bombs=M.homeDistance(s,i,targets,true,true);r.home_ignore_bricks_bombs=M.homeDistance(s,i,targets,true,false);
  const ec=s.centerCell(1);r.enemy_grid_distance=M.movementField(s,i)[ec[0]*15+ec[1]];r.phi_home_geometry_proxy=r.home_static<0?null:-r.home_static;r.phi_enemy_geometry_proxy=r.enemy_grid_distance<0?null:-r.enemy_grid_distance;
  r.home_neighbor_dynamic=[[-1,0],[1,0],[0,-1],[0,1]].map(([y,x])=>{const yy=c[0]+y,xx=c[1]+x,j=yy*15+xx;if(yy<0||yy>=13||xx<0||xx>=15||s.wall[j]||s.brick[j]||s.fuse[j]>0)return null;return M.homeDistance(s,j,targets,false,true);});
  p.step();const ac=s.centerCell(0);r.after_home_static=M.homeDistance(s,ac[0]*15+ac[1],targets);
 }
 for(const w of d.windows){const rr=d.rows.slice(w.start,w.end),bf=rr[0].home_static;w.bfs_progress=bf>=0&&rr.some(r=>r.after_home_static>=0&&r.after_home_static<bf);w.category=M.category(w);w.raw_stall=w.eligible&&w.carry&&w.net_home_progress<.1&&!w.bfs_progress;}
 d.geometry_version=2;d.geometry_amendment='GEOMETRY_CORRECTION01.json';fs.writeFileSync(path+'.tmp',JSON.stringify(d));fs.renameSync(path+'.tmp',path);
}
console.log(JSON.stringify({status:'CORRECTED',games:cfg.games.length,new_forward_calls:0}));
