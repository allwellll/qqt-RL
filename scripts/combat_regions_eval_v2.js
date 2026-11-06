'use strict';
// Frozen heavy-v6 evaluation with native respawns and passive region metrics.
// Existing replay checks remain active and verify all completed games.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),assert=require('node:assert/strict');
const entry=path.resolve('scripts/interceptor_primary_eval_defendcap4_v6.js'),read=fs.readFileSync;
function once(text,a,b){assert.equal(text.split(a).length,2,`region evaluator ABI: ${a.slice(0,65)}`);return text.replace(a,b);}
fs.readFileSync=function(file,...args){
 let text=read.call(this,file,...args);const f=path.resolve(String(file));
 if(args[0]!=='utf8')return text;
 if(f===path.resolve('web/five-item-slots-review-v1.js'))
  text=once(text,'s.bunSpawnPos=cells.map(x=>x.map(v=>v+.5));','s.bunSpawnPos=(meta.region_native_respawn_cells||cells).map(x=>x.map(v=>v+.5));');
 if(f===path.resolve('web/review_protocol.js')){
  const a='sim.bunSpawnPos=b.cells.map(a=>a.map(x=>x+.5));sim._gen++;';
  text=once(text,a,a+'if(meta.region_native_respawn_cells)sim.bunSpawnPos=meta.region_native_respawn_cells.map(a=>a.map(x=>x+.5));');
 }
 if(f===path.resolve('scripts/five_item_slots_eval.js')){
  text=once(text,"const tier=cfg.opponent_tier||'hard';","const Continuous=require('./combat_regions_metrics_v2');Continuous.install(Q);const tier=cfg.opponent_tier||'hard';");
  text=once(text,'frozen_birth:{cells:game.spawn}},level);s.__btIdentity=true;',
   'frozen_birth:{cells:game.spawn},region_native_respawn_cells:game.native_respawn},level);s.__btIdentity=true;Continuous.attach(s,game.seat);');
  text=once(text,"policy:game.mode,teacher:false,learner_seat:p,", "policy:game.mode,teacher:false,learner_seat:p,region_native_respawn_cells:game.native_respawn,start_region:game.region,relative_distance:game.distance,");
  text=once(text,"slot_by_phase:slotByPhase,", "continuous_attack:Continuous.result(s),start_region:game.region,distance:game.distance,native_respawn_cells:game.native_respawn,slot_by_phase:slotByPhase,");
 }
 return text;
};
const implementation=new Module(entry,module);implementation.filename=entry;implementation.paths=module.paths;
implementation._compile(read.call(fs,entry,'utf8'),entry);
