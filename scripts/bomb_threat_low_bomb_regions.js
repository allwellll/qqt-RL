'use strict';
// Measurement correction only: native3x3 cage center uses cell centers (+1.5).
// Reuses exact recorded actions; no forward, RNG, opponent or trajectory recollection.
const fs=require('fs'),assert=require('assert'),P=require('../web/review_protocol'),A=require('./eval_web_hunter'),{bfs}=require('./bomb_threat_low_bomb_metrics');
function fix(out){
 const m=JSON.parse(fs.readFileSync(out+'/METRICS.json')),trace=JSON.parse(fs.readFileSync(out+'/TRACE.json')),doc=JSON.parse(fs.readFileSync(out+'/review.json')),player=new P.Player(doc,A.loadLevel()),p=m.seat;
 let half=0,near=0,progress=0,rejected=0,unknown=0;const reasons={},rejects=[];const md=(a,b)=>Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1]);
 function hd(s){const [r,c]=s.centerCell(p),field=bfs(s,r*15+c),anchor=s.bunBases[s.team[p]],v=[];for(let y=anchor[0];y<anchor[0]+3;y++)for(let x=anchor[1];x<anchor[1]+3;x++){const d=field[y*15+x];if(d>=0)v.push(d);}return v.length?Math.min(...v):-1;}
 for(let i=0;i<trace.rows.length;i++){
  const s=player.sim,row=trace.rows[i],alive=s.alive[p],origin=s.bunCarried[p],pos=[s.pos[p*2],s.pos[p*2+1]],own=s.bunBases[s.team[p]].map(v=>v+1.5),enemy=s.bunBases[1-s.team[p]].map(v=>v+1.5),h=md(pos,own)<=md(pos,enemy),n=md(pos,own)<=3;
  row[8]=Number(h);row[9]=Number(n);half+=alive&&h;near+=alive&&n;const before=origin===1-s.team[p]?hd(s):-1;
  const joint=doc.actions[i],acts=[[joint[0],joint[1]===1?1:0,joint[1]===2?1:0],[joint[2],joint[3]===1?1:0,joint[3]===2?1:0]],why=require('./bomb_threat_low_bomb_metrics').rejectionSnapshot(s,acts,p);
  const pc=s.centerCell(0),p0cell=pc[0]*15+pc[1];
  assert.equal(s.t,row[0]);assert(player.step());
  if(why.requested&&!row[13]){rejected++;if(!why.why.length&&p===1&&s.lastReplayPlaced[0]&&why.cell===p0cell)why.why.push('same_cell_player0_placed_first');
    if(!why.why.length){unknown++;why.why.push('unknown');}for(const reason of why.why)reasons[reason]=(reasons[reason]||0)+1;
    rejects.push({tick:row[0],reasons:why.why,action_cell:why.cell,player0_pre_move_cell:p0cell});
  }
  if(alive&&s.alive[p]&&origin===1-s.team[p]&&s.bunCarried[p]===origin){const after=hd(s);progress+=before>=0&&after>=0&&after<before;}
 }
 assert.equal(player.sim.t,m.ticks);assert.equal(player.sim.done,true);
 m.own_half_ticks=half;m.own_cage_near_ticks=near;m.return_progress_ticks=progress;
 m.fractions={...m.fractions,own_half:half/m.ticks,own_half_alive:half/m.alive_ticks,own_cage_near:near/m.ticks,own_cage_near_alive:near/m.alive_ticks};
 m.rejected_requests=rejected;m.unexplained_rejections=unknown;m.rejection_reason_counts=reasons;m.rejected_events=rejects;
 m.category=require('./bomb_threat_low_bomb_metrics').classify(m);m.region_measurement='native3x3 cell-center+1.5 corrected v2, exact Review actions replay, no policy rerun';
 fs.writeFileSync(out+'/METRICS.json',JSON.stringify(m,null,2));fs.writeFileSync(out+'/TRACE.json',JSON.stringify(trace));
 return {index:m.index,half,near,return_progress:progress};
}
if(require.main===module)console.log(JSON.stringify(fix(process.argv[2])));
module.exports={fix};
