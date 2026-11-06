'use strict';
// Private constructed evaluator: original game/inference interface and metrics remain frozen.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const D=require('./cx_idle_observer_adapter_v1');
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const cfg=JSON.parse(fs.readFileSync(process.argv[2]));
assert.equal(process.env.CUDA_VISIBLE_DEVICES,'');
for(const [f,expected]of Object.entries(cfg.source_sha256))assert.equal(hash(f),expected,f);
D.install();
let src=fs.readFileSync('scripts/resume_gpu67_eval_v1.js','utf8');
function replace(old,newText){assert.equal(src.split(old).length-1,1,old);src=src.replace(old,newText);}
replace("'use strict';","'use strict';\nconst D=require('./cx_idle_observer_adapter_v1');");
replace("require('./interceptor_primary_inference_v1')","require('./cx_temperature_inference_v1')");
replace("left.model.greedy=game.mode==='greedy';right.model.greedy=game.opponent_mode==='greedy';", "left.model.greedy=game.mode==='greedy';right.model.greedy=game.opponent_mode==='greedy';left.model.temperature=game.temperature||1;right.model.temperature=1;");
const diagnostics="const dd=D.replay(dir+'/review.json');assert.equal(dd.r3_chain_check,true);result.idle=dd.idle;result.fountain=dd.players;result.extra_replay_hash_checks=dd.hash_checks_verified;result.temperature=game.temperature??null;result.protocol_sha256=cfg.protocol_sha256;fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(result));";
replace("fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(result));results.push(result);", "fs.writeFileSync(dir+'/RESULT.json',JSON.stringify(result));"+diagnostics+"results.push(result);");
replace("assert.equal(fileHash(dir+'/review.json'),v.review_sha256);results.push(v);continue;", "assert.equal(fileHash(dir+'/review.json'),v.review_sha256);assert.equal(v.protocol_sha256,cfg.protocol_sha256);assert(v.idle&&v.fountain&&v.extra_replay_hash_checks>0);results.push(v);continue;");
replace("module.exports={setup,threat,exits};", "module.exports={setup,threat,exits,main};");
const privateModule=new Module(path.resolve('scripts/cx_temperature_constructed_v1.js'),module);
privateModule.filename=path.resolve('scripts/cx_temperature_constructed_v1.js');privateModule.paths=module.paths;
privateModule._compile(src,privateModule.filename);
privateModule.exports.main().catch(e=>{console.error(e.stack);process.exitCode=1;});
