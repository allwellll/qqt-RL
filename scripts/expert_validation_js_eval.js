'use strict';
// Isolated adapter around the production route evaluator. The simulator and
// Hunter stay unchanged; actor and opponent receive independent seeded RNGs.
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const routeFile = path.resolve(__dirname, 'dagger_rl_route.js');

function replaceOnce(source, from, to) {
  if (source.split(from).length !== 2) throw new Error(`audit source anchor mismatch: ${from}`);
  return source.replace(from, to);
}
function replaceRange(source, start, end, replacement) {
  const i = source.indexOf(start);
  if (i < 0 || source.indexOf(start, i + start.length) >= 0) throw new Error(`audit range start mismatch: ${start}`);
  const j = source.indexOf(end, i + start.length);
  if (j < 0) throw new Error(`audit range end mismatch: ${end}`);
  return source.slice(0, i) + replacement + source.slice(j + end.length);
}

function build(mode) {
  if (!['categorical', 'greedy'].includes(mode)) throw new Error('explicit action-selection mode required');
  let source = fs.readFileSync(routeFile, 'utf8');
  source = replaceOnce(source, "require('./dagger_rl_metrics')", "require('./ppo_scale_item_metrics_v4')");
  source = replaceRange(source,
    '  const originalForward=loaded.model.forward.bind(loaded.model);let lastLogits=null;',
    '  const selected=process.env.JE10_ROUTE_TASK?',
    `  loaded.model.greedy=${mode === 'greedy'};
  const originalForward=loaded.model.forward.bind(loaded.model);let lastLogits=null;
  loaded.model.forward=async(...args)=>{const z=await originalForward(...args);lastLogits=z;return z;};
  const originalAct=loaded.model.act.bind(loaded.model);
  loaded.model.act=async(s,pid,rng)=>{
    const result=await originalAct(s,pid,rng),masks=s.legalMask(),abilityMask=loaded.model._abilityMask(s,masks,pid);
    const moveLogits=Array.from(lastLogits.move),abilityLogits=Array.from(lastLogits.bomb);
    const probabilities=(logits,legal)=>{const mx=Math.max(...logits.filter((_,i)=>legal[i]));const weights=logits.map((v,i)=>legal[i]?Math.exp(v-mx):0);const den=weights.reduce((a,b)=>a+b,0);return weights.map(v=>v/den);};
    const moveProbabilities=probabilities(moveLogits,masks.mm[pid]),abilityProbabilities=probabilities(abilityLogits,abilityMask);
    const entropy=p=>-p.reduce((sum,v)=>sum+(v>0?v*Math.log(v):0),0);
    const greedyIndex=(logits,legal)=>{let best=-1;for(let i=0;i<logits.length;i++)if(legal[i]&&(best<0||logits[i]>logits[best]))best=i;return best;};
    const greedyMove=greedyIndex(moveLogits,masks.mm[pid]),greedyAbility=greedyIndex(abilityLogits,abilityMask);
    s._expertActStats=s._expertActStats||{};s._expertActStats[pid]={moveProbabilities,abilityProbabilities,
      moveEntropy:entropy(moveProbabilities),abilityEntropy:entropy(abilityProbabilities),greedyMove,greedyAbility};
    s._daggerItemDecision=s._daggerItemDecision||{};s._daggerItemDecision[pid]={legal:Array.from(abilityMask),logits:abilityLogits,probabilities:abilityProbabilities,greedy:greedyAbility};
    return result;
  };
  const selected=process.env.JE10_ROUTE_TASK?`);
  source = replaceOnce(source, "policy:'greedy'", `policy:'${mode}'`);
  source = replaceOnce(source,
    'const rng=Q.mulberry32(seed^0x515154),hunter=authority.createHunter(\'normal\');',
    'const actorRng=Q.mulberry32(seed^0x515154),opponentRng=Q.mulberry32(seed^0x7a11c0de),hunter=authority.createHunter(\'normal\');');
  source = replaceOnce(source,
    'const stageCounts={},firstByPhase={},itemMetrics=new ItemMetrics(s,team);',
    'const stageCounts={},firstByPhase={},actionStats={},itemMetrics=new ItemMetrics(s,team);');
  source = replaceOnce(source,
    'const a=await loaded.model.act(s,team,rng),bot=hunter.adapter.act({schema:\'qqt.bot.observation/v1\',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,rng);',
    `const a=await loaded.model.act(s,team,actorRng),bot=hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:s.t,state:null,legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim:s}},enemy,opponentRng);
        const phaseKey=obj?.phase||(s.bunCarried[team]>=0?'C_RETURN':window?'WINDOW_NO_TARGET':'COMBAT');
        const exposure=actionStats[phaseKey]||(actionStats[phaseKey]={ticks:0,move_entropy_sum:0,ability_entropy_sum:0,move_action_counts:[0,0,0,0,0],ability_action_counts:[0,0,0]});
        const actStats=s._expertActStats?.[team];if(actStats){exposure.ticks++;exposure.move_entropy_sum+=actStats.moveEntropy;exposure.ability_entropy_sum+=actStats.abilityEntropy;
          exposure.move_action_counts[a[0]]++;exposure.ability_action_counts[a[2]?2:a[1]?1:0]++;}`);
  source = replaceOnce(source,
    'const ep={...control.metrics',
    `itemMetrics.flush('evaluation_end');const ep={item_measurement_version:4,item_chain_events:itemMetrics.chainEvents,action_selection:'${mode}',action_exposure:actionStats,...control.metrics`);
  source = replaceOnce(source,
    'module.exports={continuous,objective,setup,TASKS};',
    'module.exports={run,continuous,objective,setup,TASKS};');
  const mod = new Module(routeFile, module);
  mod.filename = routeFile;
  mod.paths = module.paths;
  mod._compile(source, routeFile);
  return mod.exports;
}

if (require.main === module) {
  const [mode, model, output, regions] = process.argv.slice(2);
  build(mode).run(model, output, regions).catch(error => { console.error(error.stack || error); process.exitCode = 1; });
}
module.exports = { build };
