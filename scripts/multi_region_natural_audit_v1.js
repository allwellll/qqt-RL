'use strict';
// Replay-only CPU audit: no inference, action selection, training or writes to
// source reviews. Native causal attribution is replayed and kept separate.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto');
const Q=require('../web/sim-bun-native-v2');
const file=path.resolve('web/five-item-slots-review-v1.js');
const text=fs.readFileSync(file,'utf8'),before='return s;\n}';
if(text.split(before).length!==2)throw Error('review setup ABI changed');
const loaded=new Module(file,module);loaded.filename=file;loaded.paths=module.paths;
loaded._compile(text.replace(before,'require("../scripts/interceptor_defend_cap4_v1").applyReplayCapacity(s,meta);return s;\n}'),file);
const P=loaded.exports,Source=require('./combat_specialist_g7_source_metrics');Source.install(Q);
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function region(y,x){if(y>=4&&y<9&&x>=5&&x<10)return 'center';return (y<6?'top_':'bottom_')+(x<7?'left':'right');}
const empty=()=>({alive_ticks:0,actionable_ticks:0,near_enemy_ticks:0,combat_proxy_ticks:0,causal_controls:0,causal_kills:0,physical_self_control:0,deaths:0});
async function audit(f){
 const bytes=fs.readFileSync(f),d=JSON.parse(bytes),level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json','utf8')).find(x=>x.qqt_id===806);
 const player=new P.Player(d,level);await player.verify();player.reset();
 const p=d.meta.learner_seat,q=1-p,groups=Object.fromEntries(['top_left','top_right','bottom_left','bottom_right','center'].map(k=>[k,empty()]));
 Source.attach(player.sim,{learner:p});let lastPlaced=-1000;const regions=new Map();
 const spawn=Array.from(player.sim.pos);
 while(player.sim.t<d.actions.length){
  const s=player.sim,g=groups[region(s.pos[2*p],s.pos[2*p+1])],alive=s.alive[p],active=alive&&s.trapped[p]===0;
  const distance=Math.abs(s.pos[2*p]-s.pos[2*q])+Math.abs(s.pos[2*p+1]-s.pos[2*q+1]);
  g.alive_ticks+=Number(alive);g.actionable_ticks+=Number(active);g.near_enemy_ticks+=Number(active&&s.alive[q]&&s.trapped[q]===0&&distance<=4);
  g.combat_proxy_ticks+=Number(active&&s.alive[q]&&distance<=6&&s.t-lastPlaced<=31);
  player.step();if(player.info.placed?.some(Boolean))lastPlaced=player.sim.t;
  regions.set(player.sim.t,[region(player.sim.pos[0],player.sim.pos[1]),region(player.sim.pos[2],player.sim.pos[3])]);
 }
 const source=Source.exportMetrics(player.sim);
 for(const c of source.controls){const r=regions.get(c.tick);if(!r)continue;
  if(c.victim===q&&c.causal_kind==='unique_other')groups[r[p]].causal_controls++;
  if(c.victim===p&&c.physical_kind==='unique_self')groups[r[p]].physical_self_control++;
 }
 for(const death of source.deaths){const r=regions.get(death.tick);if(death.victim===p&&r)groups[r[p]].deaths++;
  const initial=death.initial_control,at=initial&&regions.get(initial.tick);
  if(death.victim===q&&initial?.causal_kind==='unique_other'&&at)groups[at[p]].causal_kills++;
 }
 return {review:f,review_sha256:sha(bytes),seed:d.meta.seed,seat:p,mode:d.meta.policy,checkpoint:d.meta.checkpoint,
  frozen_birth:d.meta.frozen_birth,spawn,spawn_side:spawn[2*p+1]<7.5?'left':'right',complete:player.sim.done,winner:player.sim.winner,
  verified:true,groups,source_missing:source.missing_data};
}
async function main(){const config=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),rows=[];
 for(const f of config.reviews){rows.push(await audit(f));fs.writeFileSync(config.output,JSON.stringify({status:'IN_PROGRESS',rows},null,2));}
 fs.writeFileSync(config.output,JSON.stringify({status:'COMPLETE',rows,definitions:{region:'center first, then quadrant by row6/col7; zero samples reported',combat:'observational proxy: both alive, own actionable, distance<=6, actual bubble in preceding31 ticks',causal_kill:'unique causal other-source at initial control, region of attacker at initial control; no intent claim'},limitations:['first fixed reviews only; not population confirmation','combat proxy is not an advantage or reward','no-region/no-contact rows cannot establish region capability']},null,2));}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
