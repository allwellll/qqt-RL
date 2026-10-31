'use strict';
// Run frozen production inference unchanged, adding only requested missing counters.
const fs=require('fs'),path=require('path'),Module=require('module');
const {PhaseSafety}=require('./combat_item_phase_safety');
const filename=path.join(__dirname,'combat_item_route.js');
let source=fs.readFileSync(filename,'utf8');
function replace(a,b){if(source.split(a).length!==2)throw Error('frozen instrumentation anchor mismatch');source=source.replace(a,b);}
replace('const stageCounts={},firstByPhase={},itemMetrics=new ItemMetrics(s,team);','const phaseSafety=new PhaseSafety(team);const stageCounts={},firstByPhase={},itemMetrics=new ItemMetrics(s,team);');
replace('const a=await loaded.model.act(s,team,rng),bot=', 'phaseSafety.before(s);const a=await loaded.model.act(s,team,rng),bot=');
replace('m.pickup+=Number(picked);','phaseSafety.after(s,info,picked);m.pickup+=Number(picked);');
replace('item_metrics:itemMetrics.rows,','phase_safety:phaseSafety.rows,pre_first_acquisition_deaths:phaseSafety.pre_first_acquisition_deaths,item_metrics:itemMetrics.rows,');
source=source.replace("const {StockAcquisitionFunnel}=require('./stock_acquisition_funnel');", "const {StockAcquisitionFunnel}=require('./stock_acquisition_funnel');const {PhaseSafety}=require('./combat_item_phase_safety');");
const m=new Module(filename,module);m.filename=filename;m.paths=module.paths;
// Frozen CLI invokes main() only for require.main; explicitly call its exported runner.
source+='\nmodule.exports.run=run;\n';m._compile(source,filename);
m.exports.run(...process.argv.slice(2)).catch(e=>{console.error(e);process.exitCode=1;});
