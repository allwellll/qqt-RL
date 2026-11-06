'use strict';
// CPU-only, verified action replay. No policy decisions or simulator changes.
const fs=require('node:fs'),Module=require('node:module'),path=require('node:path');
const file=path.resolve('scripts/multi_region_natural_audit_v1.js');
let code=fs.readFileSync(file,'utf8');
const replace=(before,after)=>{if(code.split(before).length!==2)throw Error('audit ABI mismatch');code=code.replace(before,after);};
replace('deaths:0});','deaths:0,legal_capacity_ticks:0,legal_repeat_opportunity_ticks:0,placed:0,repeat_within30:0,placement_gaps:[],legal_opportunity_gaps:[],active_approach_ticks:0});');
replace('let lastPlaced=-1000;const regions=new Map();','let lastPlaced=-1000,lastOwn=-1,eligibleSinceLast=0;const regions=new Map();');
replace('  player.step();if(player.info.placed?.some(Boolean))lastPlaced=player.sim.t;',`  const cell=s.centerCell(p),index=cell[0]*15+cell[1];
  // Fuse is decremented before placement in the actual native step.
  const legal=active&&s.bunCarried[p]<0&&!s.brick[index]&&s.fuse[index]<=1&&s.liveBombs(p)<s.bombsCap[p];
  g.legal_capacity_ticks+=Number(legal);
  if(lastOwn>=0&&legal){eligibleSinceLast++;g.legal_repeat_opportunity_ticks++;}
  player.step();if(player.info.placed?.some(Boolean))lastPlaced=player.sim.t;
  const next=player.sim,afterDistance=Math.abs(next.pos[2*p]-next.pos[2*q])+Math.abs(next.pos[2*p+1]-next.pos[2*q+1]);
  g.active_approach_ticks+=Number(active&&s.alive[q]&&distance<=6&&afterDistance<distance-1e-6);
  if(player.info.placed?.[p]){g.placed++;if(lastOwn>=0){g.placement_gaps.push(next.t-lastOwn);g.legal_opportunity_gaps.push(eligibleSinceLast);g.repeat_within30+=Number(next.t-lastOwn<=30);}lastOwn=next.t;eligibleSinceLast=0;}
  if(player.info.died?.[p]){lastOwn=-1;eligibleSinceLast=0;}`);
replace("if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});","module.exports={audit};");
const loaded=new Module(file,module);loaded.filename=file;loaded.paths=module.paths;loaded._compile(code,file);
async function main(){
 const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),rows=[];
 for(const entry of input.reviews){rows.push({...await loaded.exports.audit(entry.review),role:entry.role});
  fs.writeFileSync(input.output,JSON.stringify({status:'RUNNING',rows},null,2));}
 fs.writeFileSync(input.output,JSON.stringify({status:'COMPLETE',rows,
  definitions:{legal_capacity:'native next-step legal placement with available capacity, actionable and not carrying',
   repeat_within30:'actual own placements <=30 ticks apart, reset after own death; not bubble-count skill proxy',
   legal_opportunity_gap:'eligible placement ticks between own placements; capacity/carry blocked ticks excluded',
   approach:'both alive/actionable, <=6 Manhattan cells, distance decreases in the recorded action tick',
   controls_and_kills:'strict native unique-other source at initial control, reported separately from placements'},
  limitations:['fixed first four seeds per seat/mode; no outcome selection; low sample observational comparison',
   'legal opportunity does not prove tactically safe placement',
   'control/kill per placement is a descriptive ratio, not a matched individual-bubble conversion probability',
   'primary evaluations use frozen heavy-v6 opponent, not light-v2 opponent; learner checkpoints correctly labeled']},null,2));
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
