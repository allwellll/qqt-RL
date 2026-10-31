'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),Module=require('node:module'),path=require('node:path');
const {build}=require('../scripts/expert_validation_js_eval');
for(const tier of ['easy','normal','hard'])for(const mode of ['greedy','categorical'])assert.equal(typeof build(mode,tier).run,'function');
assert.throws(()=>build('greedy','unknown'),/difficulty/);
// Intercept only the real Hunter factory, run zero episodes, and inspect the
// compiled route module to verify default=normal and requested tier binding.
const file=path.resolve('scripts/expert_validation_js_eval.js');
const src=fs.readFileSync(file,'utf8').replace('mod._compile(source, routeFile);','mod._compile(source, routeFile); mod.exports.__source=source;');
const m=new Module(file,module);m.filename=file;m.paths=module.paths;m._compile(src,file);
assert.equal(m.exports.build('greedy').__source,m.exports.build('greedy','normal').__source);
for(const tier of ['easy','normal','hard'])assert.ok(m.exports.build('greedy',tier).__source.includes(`hunter=authority.createHunter('${tier}')`));
console.log('fresh256: default source exactly normal; each tier bound to actual Hunter, modes preserved');
