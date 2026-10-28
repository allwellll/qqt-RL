'use strict';
// Isolated native-JS natural audit. Production deterministic evaluation is intact.
const fs=require('fs'),path=require('path'),Module=require('module');
const sourceFile=path.join(__dirname,'dagger_rl_route.js');
function rewrite(source,from,to){
  if(source.split(from).length!==2)throw Error(`audit adapter anchor mismatch: ${from}`);
  return source.replace(from,to);
}
function adapter(mode){
  if(!['categorical','greedy'].includes(mode))throw Error('explicit action-selection mode required');
  let source=fs.readFileSync(sourceFile,'utf8');
  source=rewrite(source,"require('./dagger_rl_metrics')","require('./ppo_scale_item_metrics_v4')");
  source=rewrite(source,'  // Observe the existing forward/act calls;',
    `  loaded.model.greedy=${mode==='greedy'};\n  // Observe the existing forward/act calls;`);
  source=rewrite(source,'      const ep={...control.metrics',
    "      itemMetrics.flush('evaluation_end');\n      const ep={item_measurement_version:4,item_chain_events:itemMetrics.chainEvents,action_selection:'"+mode+"',...control.metrics");
  source=rewrite(source,"policy:'greedy'",`policy:'${mode}'`);
  source=rewrite(source,'module.exports={continuous,objective,setup,TASKS};',
    'module.exports={run,continuous,objective,setup,TASKS};');
  const m=new Module(sourceFile,module);m.filename=sourceFile;m.paths=module.paths;
  m._compile(source,sourceFile);return m.exports;
}
if(require.main===module){
  const [mode,...args]=process.argv.slice(2);
  adapter(mode).run(...args).catch(e=>{console.error(e);process.exitCode=1;});
}
module.exports={adapter};
