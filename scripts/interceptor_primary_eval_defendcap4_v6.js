'use strict';
// Independent frozen full-interceptor primary evaluation. Historical entries remain.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),assert=require('node:assert/strict');
// CPU-only per-evaluation setup; original simulator and review files unchanged.
const Capacity=require('./interceptor_defend_cap4_v1');
for(const relative of ['web/review_protocol.js','web/five-item-slots-review-v1.js']){
 const f=path.resolve(relative);let text=fs.readFileSync(f,'utf8');
 if(relative==='web/review_protocol.js'){
  const before='sim.bunSpawnPos=b.cells.map(a=>a.map(x=>x+.5));sim._gen++;';
  assert.equal(text.split(before).length,2);
  text=text.replace(before,'sim.bunSpawnPos=b.cells.map(a=>a.map(x=>x+.5));require("../scripts/interceptor_defend_cap4_v1").applyReplayCapacity(sim,meta);sim._gen++;');
 }else{
  const before='return s;\n}';assert.equal(text.split(before).length,2);
  text=text.replace(before,'require("../scripts/interceptor_defend_cap4_v1").applyReplayCapacity(s,meta);return s;\n}');
 }
 const loaded=new Module(f,module);loaded.filename=f;loaded.paths=Module._nodeModulePaths(path.dirname(f));loaded._compile(text,f);require.cache[f]=loaded;
}

const file=path.resolve('scripts/five_item_slots_eval.js');let source=fs.readFileSync(file,'utf8');
function replace(before,after){assert.equal(source.split(before).length,2,`evaluation ABI changed: ${before.slice(0,70)}`);source=source.replace(before,after);}
replace("const {loadModel}=require('./five_item_slots_inference');","const {loadModel}=require('./interceptor_primary_inference_v1');");
replace("const {snapshot,FunnelTracker}=require('./r4_iteration_metrics_20261101_funnel_helper');","const {snapshot,FunnelTracker}=require('./r4_iteration_metrics_20261101_funnel_helper');\nconst Source=require('./combat_specialist_g7_source_metrics');Source.install(Q);");
replace("const bot=Bots.createDefaultRegistry({hunter:Hunter}).create('bun.hunter',{difficulty:tier});bot.reset({seed:game.seed^0xB07});",`const Native=require('./interceptor_defend_cap4_bot_v6');assert.equal(cfg.opponent,Native.VERSION);
  const native=new Native.BunInterceptorBot({allowFallback:true});
  const bot={reset:c=>native.reset(c.seed),act:(input,pid)=>{const a=native.act(input.metadata.sim,pid);return {move:a[0],ability:a[1],itemSlot:0};}};bot.reset({seed:game.seed^0xB07});Source.attach(s,{learner:p});`);
replace("const p=game.seat,q=1-p,model=loaded.model;model.greedy=game.mode==='greedy';", "const p=game.seat,q=1-p,model=loaded.model;model.greedy=game.mode==='greedy';Capacity.applyInitialCapacity(s,q,4);");
replace("policy:game.mode,teacher:false,learner_seat:p,", "policy:game.mode,teacher:false,learner_seat:p,defender_player:q,defender_initial_bomb_capacity:4,defender_death_floor_bombs:s.loBombs[q],");
replace("z.legal+=d.ability_mask[i+2];z.probability_sum+=d.ability_probability[i+2];","z.legal+=Number(d.ability_mask[i+2]||false);z.probability_sum+=d.ability_probability[i+2]||0;");
replace("model:'five-slot technical acceptance'","model:cfg.display_name||'interceptor primary carrying evaluation',source_policy_ability_actions:loaded.meta.ability_actions");
replace("opponent_sha256:hash(fs.readFileSync('web/bun_hunter_bot.js'))","opponent_sha256:hash(fs.readFileSync('scripts/interceptor_defend_cap4_bot_v6.js'))");
replace("opponent:`production JS Hunter/${tier}`,opponent_tier:tier","opponent:cfg.opponent,opponent_tier:null");
replace("const result={status:'PASS',seed:game.seed","const sourceMetrics=Source.exportMetrics(s);fs.writeFileSync(dir+'/SOURCE_EVENTS.json',JSON.stringify(sourceMetrics));\n  const result={opponent:cfg.opponent,engine:'JS',source_policy_ability_actions:loaded.meta.ability_actions,source_metrics:sourceMetrics,source_events_sha256:hash(fs.readFileSync(dir+'/SOURCE_EVENTS.json')),status:'PASS',seed:game.seed");
replace("source_metrics:sourceMetrics,source_events_sha256:","primary_custody:require('./interceptor_primary_chain_metrics_v1').summarize(sourceMetrics,R3.exportMetrics(s),p),source_metrics:sourceMetrics,source_events_sha256:");
replace("if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});","main().catch(e=>{console.error(e.stack);process.exitCode=1;});");
source="const Capacity=require('./interceptor_defend_cap4_v1');\n"+source;
const implementation=new Module(file,module);implementation.filename=file;implementation.paths=module.paths;implementation._compile(source,file);
