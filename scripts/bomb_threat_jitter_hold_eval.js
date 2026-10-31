'use strict';
process.env.QQT_STOCK_RULE='stock-window-v1';process.env.BT_FRESH_TIER='hard';
const fs=require('fs'),assert=require('assert'),{holdController}=require('./bomb_threat_jitter_hold'),M=require('./bomb_threat_p0_jitter_metrics');
const B='runs/bomb_threat_v1/jitter_followup_u800',cfg=JSON.parse(fs.readFileSync(B+'/PROTOCOL.json'));
const [arm,startText,endText]=process.argv.slice(2),start=Number(startText),end=Number(endText),period=cfg.inference_arms[arm];assert(period);
const out=`${B}/${arm}/shard_${String(start).padStart(3,'0')}.json`;fs.mkdirSync(`${B}/${arm}`,{recursive:true});
const Q=require('../web/sim-stock-window-v1'),inference=require('./je05_web_hunter'),load=inference.loadModel,hold=holdController(period);let modelsha;
inference.loadModel=async(...args)=>{const x=await load(...args);assert.equal(x.sha,cfg.model.model_sha256);modelsha=x.sha;const act=x.model.act.bind(x.model);x.model.act=async(s,pid,rng)=>hold.apply(s,pid,await act(s,pid,rng));return x;};
const step=Q.Sim.prototype.step,episodes=new Map();
Q.Sim.prototype.step=function(actions){
 if(!this.__btIdentity)return step.call(this,actions);
 const s=this,p=s.__btIdentity.seat,enemy=1-p,seed=s.__btIdentity.seed;
 if(!episodes.has(seed))episodes.set(seed,{seed,seat:p,rows:[]});const rows=episodes.get(seed).rows,anchor=s.bunBases[s.team[p]],hc=anchor.map(v=>v+1.5),targets=[];
 for(let y=anchor[0];y<anchor[0]+3;y++)for(let x=anchor[1];x<anchor[1]+3;x++)targets.push(y*15+x);
 const cell=s.centerCell(p),pos=s.pos.slice(p*2,p*2+2),controlled=!!s.alive[p]&&!s.trapped[p];
 const r={tick:s.t,pos,controlled,carry:controlled&&s.bunCarried[p]===s.team[enemy],move:actions[p][0],enemy_move:actions[enemy][0],enemy_controlled:!!s.alive[enemy]&&!s.trapped[enemy],home_static:M.homeDistance(s,cell[0]*15+cell[1],targets),home_center_distance:Math.hypot(pos[0]-hc[0],pos[1]-hc[1])};
 const z=step.call(s,actions);r.after=s.pos.slice(p*2,p*2+2);r.next_controlled=!!s.alive[p]&&!s.trapped[p];r.next_carry=r.next_controlled&&s.bunCarried[p]===s.team[enemy];const ac=s.centerCell(p);r.after_home_static=M.homeDistance(s,ac[0]*15+ac[1],targets);r.after_home_center_distance=Math.hypot(r.after[0]-hc[0],r.after[1]-hc[1]);rows.push(r);return z;
};
process.env.JE10_ROUTE_TASK='natural_full2400';process.env.JE10_ROUTE_GAMES='64';process.env.JE10_ROUTE_START=String(start);process.env.JE10_ROUTE_END=String(end);process.env.JE10_ROUTE_SEED_START=String(cfg.seed_start);
process.argv=[process.argv[0],process.argv[1],'greedy',cfg.model.model_json,out,'runs/ppo_scale_20261025/r1_longrun/cpu_eval_inputs/regions.json',cfg.spawn_manifest];
require('./bomb_threat_fresh256_eval');
process.on('beforeExit',()=>{
 if(!fs.existsSync(out+'.outcome.json')||fs.existsSync(out+'.diagnostic.json'))return;
 const outcomes=JSON.parse(fs.readFileSync(out+'.outcome.json')).episodes,metrics=[];
 for(const ep of outcomes){const rr=episodes.get(ep.seed).rows;assert(rr.length===ep.ticks&&ep.done);let stall=0,osc=0;
  for(let i=0;i+50<=rr.length;i+=50){const rows=rr.slice(i,i+50),a=rows[0],z=rows.at(-1),k=M.kinematics([...rows.map(r=>r.pos),z.after]);const eligible=rows.every(r=>r.controlled&&r.next_controlled),car=rows.every(r=>r.carry&&r.next_carry),bp=a.home_static>=0&&rows.some(r=>r.after_home_static>=0&&r.after_home_static<a.home_static);if(eligible&&car&&a.home_center_distance-z.after_home_center_distance<.1&&!bp)stall+=50;if(eligible&&k.oscillation)osc+=50;}
  const rev=M.reversals(rr.map(r=>r.move),rr.map(r=>r.controlled)),bot=M.reversals(rr.map(r=>r.enemy_move),rr.map(r=>r.enemy_controlled));
  metrics.push({...ep,move_ticks:rr.filter(r=>r.controlled&&r.move<4).length,lr_reversals:rev.events.length,reversal_intervals:rev.intervals,enemy_move_ticks:rr.filter(r=>r.enemy_controlled&&r.enemy_move<4).length,enemy_lr_reversals:bot.events.length,enemy_reversal_intervals:bot.intervals,carry_stall_ticks:stall,oscillation_ticks:osc,carry_ticks:rr.filter(r=>r.carry).length});
 }
 fs.writeFileSync(out+'.trace.json',JSON.stringify({arm,period,model_sha256:modelsha,episodes:[...episodes.values()]}));fs.writeFileSync(out+'.diagnostic.json',JSON.stringify({arm,period,model_sha256:modelsha,hold:hold.stats(),episodes:metrics}));console.log(JSON.stringify({arm,start,end,games:metrics.length,hold:hold.stats()}));
});
