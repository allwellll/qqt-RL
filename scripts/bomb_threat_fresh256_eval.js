'use strict';
// Isolated full-game adapter: retain frozen births/outcome accounting, only
// substitute the explicitly preregistered production Hunter difficulty.
const fs=require('node:fs'),Module=require('node:module'),path=require('node:path');
const tier=process.env.BT_FRESH_TIER;
if(!['hard','normal','easy'].includes(tier))throw Error('explicit fresh Hunter tier');
const file=path.resolve('scripts/bomb_threat_eval.js');
const source=fs.readFileSync(file,'utf8');
const from="require('./expert_validation_js_eval').build(mode)";
if(source.split(from).length!==2)throw Error('fresh adapter source anchor');
const m=new Module(file,module);m.filename=file;m.paths=module.paths;
m._compile(source.replace(from,`require('./expert_validation_js_eval').build(mode,${JSON.stringify(tier)})`),file);
