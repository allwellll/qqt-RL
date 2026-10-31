'use strict';
// Original Review actions only. CPU forwards never choose or change an action.
process.env.QQT_STOCK_RULE='stock-window-v1';
const fs=require('fs'),crypto=require('crypto'),assert=require('node:assert/strict'),P=require('../web/review_protocol'),A=require('./eval_web_hunter');
const M=require('./bomb_threat_p0_jitter_metrics'),{bfs}=require('./bomb_threat_low_bomb_metrics'),{probabilities}=require('./bomb_threat_iter2_actions');
const B='runs/bomb_threat_v1/p0_jitter_u800',cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json')),hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const atomic=(f,x)=>{fs.writeFileSync(f+'.tmp',JSON.stringify(x));fs.renameSync(f+'.tmp',f);};
async function main(){
 const {model,sha}=await require('./je05_web_hunter').loadModel(cfg.model.model_json);assert.equal(sha,cfg.model.model_sha256);
 for(const index of process.argv.slice(2).map(Number)){
  const dest=`${B}/game_${String(index).padStart(3,'0')}.json`;if(fs.existsSync(dest))continue;
  const e=cfg.games.find(g=>g.index===index),out=`${cfg.source}/game_${String(index).padStart(3,'0')}`,source=JSON.parse(fs.readFileSync(out+'/COMPLETE.json'));
  assert.equal(hash(out+'/TRACE.json'),source.trace_sha256);assert.equal(hash(out+'/review.json'),source.review_sha256);
  const doc=JSON.parse(fs.readFileSync(out+'/review.json')),tr=JSON.parse(fs.readFileSync(out+'/TRACE.json')),p=new P.Player(doc,A.loadLevel());await p.verify();p.reset();
  const rows=[],physics=[];let mismatches=0,abilityMismatch=0;const spawn=p.sim.pos.slice(2,4);
  const original=p.sim._steer,originalTry=p.sim._tryMove;
  p.sim._steer=function(y,x,mv,blocked,dist,pid){
   if(pid===0){const proj=z=>mv===0?y-z[0]:mv===1?z[0]-y:mv===2?x-z[1]:z[1]-x;
    const wa=Uint8Array.from(this.wall),br=Uint8Array.from(this.wall,(v,i)=>Number(v||this.brick[i]));
    const a=proj(originalTry.call(this,y,x,mv,wa,dist)),b=proj(originalTry.call(this,y,x,mv,br,dist)),c=proj(originalTry.call(this,y,x,mv,blocked,dist));
    physics[this.t]={dist,requested_axis:c,wall_limited:a<dist*.95,brick_limited:b<a-1e-6,bomb_limited:c<b-1e-6};
   }return original.call(this,y,x,mv,blocked,dist,pid);
  };
  for(let t=0;t<doc.actions.length;t++){
   const s=p.sim,r=tr.rows[t];assert.equal(s.t,t);assert.equal(s.pos[0],r[24][0]);assert.equal(s.pos[1],r[24][1]);assert.equal(s.bunCarried[0],r[10]);
   const pos=s.pos.slice(0,2),enemy=s.pos.slice(2,4),cell=s.centerCell(0),i=cell[0]*15+cell[1],own=s.bunBases[s.team[0]],ec=s.bunBases[1-s.team[0]].map(x=>x+1.5),hc=own.map(x=>x+1.5),targets=[];
   for(let y=own[0];y<own[0]+3;y++)for(let x=own[1];x<own[1]+3;x++)targets.push(y*15+x);
   const input=model._getLaggedInputs(s,0);input.state=require('./je05_bun_encoding').encodeState(s,0);
   const f=await model.forward(input.obs,input.state),mask=s.legalMask(),am=model._abilityMask(s,mask,0),top=(v,m)=>Array.from(v,(_,i)=>i).filter(i=>m[i]).sort((a,b)=>v[b]-v[a]||a-b)[0];
   const mt=top(f.move,mask.mm[0]),raw=top(f.move,[1,1,1,1,1]),at=top(f.bomb,am),action=doc.actions[t][0];mismatches+=mt!==action;abilityMismatch+=at!==doc.actions[t][1];
   const hd=M.homeDistance(s,i,targets),dy=M.homeDistance(s,i,targets,false,true),ib=M.homeDistance(s,i,targets,true,true),nb=M.homeDistance(s,i,targets,true,false),ed=M.movementField(s,i)[s.centerCell(1)[0]*15+s.centerCell(1)[1]];
   const md=(a,b)=>Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1]);const region=md(pos,ec)<=3?'敌笼附近':md(pos,spawn)<=2?'敌方出生点':'其他';
   const controlled=!!s.alive[0]&&s.trapped[0]===0,carry=controlled&&s.bunCarried[0]===1-s.team[0],slide=s.movementStatus[0];
   const neighbor=[[-1,0],[1,0],[0,-1],[0,1]].map(([y,x])=>{const q=(cell[0]+y)*15+cell[1]+x;if(cell[0]+y<0||cell[0]+y>=13||cell[1]+x<0||cell[1]+x>=15||s.wall[q]||s.brick[q]||s.fuse[q]>0)return null;return M.homeDistance(s,q,targets,false,true);});
   const row={tick:t,pos:Array.from(pos),enemy_pos:Array.from(enemy),controlled,carry,move:action,forced_status:slide,cell:i,region,enemy_grid_distance:ed,enemy_euclidean:Math.hypot(pos[0]-enemy[0],pos[1]-enemy[1]),home_center_distance:Math.hypot(pos[0]-hc[0],pos[1]-hc[1]),home_static:hd,home_dynamic:dy,home_ignore_bricks_live_bombs:ib,home_ignore_bricks_bombs:nb,home_neighbor_dynamic:neighbor,phi_home_geometry_proxy:hd<0?null:-hd,phi_enemy_geometry_proxy:ed<0?null:-ed,grid_boundary_distance:Math.min(...pos.map(x=>Math.min(x%1,1-x%1))),move_logits:Array.from(f.move),move_mask:Array.from(mask.mm[0]),move_prob:probabilities(Array.from(f.move),mask.mm[0]),move_top:mt,raw_move_top:raw,bomb_legal:r[15],bomb_probability:r[16],enemy_distance:r[5]};
   assert(p.step());row.after=Array.from(p.sim.pos.slice(0,2));row.after_home_center_distance=Math.hypot(row.after[0]-hc[0],row.after[1]-hc[1]);const ac=p.sim.centerCell(0);row.after_home_static=M.homeDistance(p.sim,ac[0]*15+ac[1],targets);row.next_controlled=!!p.sim.alive[0]&&p.sim.trapped[0]===0;row.next_carry=row.next_controlled&&p.sim.bunCarried[0]===1-p.sim.team[0];row.displacement=Math.hypot(row.after[0]-pos[0],row.after[1]-pos[1]);row.physics=physics[t]||null;rows.push(row);
  }
  assert.equal(mismatches,0,'original move argmax consistency');assert.equal(abilityMismatch,0,'original ability argmax consistency');assert(p.sim.done);
  const windows=[];
  for(let start=0;start+50<=rows.length;start+=50){const rr=rows.slice(start,start+50),z=M.kinematics([...rr.map(r=>r.pos),rr.at(-1).after]),eligible=rr.every(r=>r.controlled&&r.next_controlled),carry=rr.every(r=>r.carry&&r.next_carry),legal=rr.filter(r=>r.bomb_legal),bf=rr[0].home_static;
   const w={start,end:start+50,...z,eligible,carry,net_home_progress:rr[0].home_center_distance-rr.at(-1).after_home_center_distance,bfs_progress:bf>=0&&rr.some(r=>r.after_home_static>=0&&r.after_home_static<bf),legal:legal.length,prob:legal.length?legal.reduce((s,r)=>s+r.bomb_probability,0)/legal.length:null};w.category=M.category(w);w.raw_oscillation=eligible&&z.oscillation;w.raw_stall=eligible&&carry&&w.net_home_progress<.1&&!w.bfs_progress;windows.push(w);
  }
  const rev=M.reversals(rows.map(r=>r.move),rows.map(r=>r.controlled));
  atomic(dest,{status:'MATCH',index,seed:e.seed,group:e.group,ticks:rows.length,source_sha256:source,model_sha256:sha,forward_parity_max_error:model.forwardMaxError,move_argmax_mismatches:mismatches,ability_argmax_mismatches:abilityMismatch,rows,windows,reversals:rev,terminal:{tick:p.sim.t,done:p.sim.done,winner:p.sim.winner,stored:p.sim.bunStored},tail_ticks:rows.length%50});
  atomic(`${B}/progress_${process.argv[2]}.json`,{last_index:index,unix:Date.now()/1000});console.log(JSON.stringify({index,ticks:rows.length,windows:windows.length,mismatches}));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
