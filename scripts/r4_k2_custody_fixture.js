'use strict';
// Diagnostic only: keep the original physical action trace and replace only
// the independently computed scalar potential oracle for the new K2 version.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
let source=fs.readFileSync('scripts/bun_native_v2_custody_fixture.js','utf8');
const replacements=[
 ['return -.30-.50*Math.max(0,Math.min(1,1-f[r*15+c]/span));','return -.40-.80*Math.max(0,Math.min(1,1-f[r*15+c]/span));'],
 ['return target<0?-.8:-.10-.10*Math.max(0,Math.min(1,1-field([target])[r*15+c]/span));','return target<0?-1.20:-.10;']
];
for(const [a,b] of replacements){if(source.split(a).length!==2)throw Error('fixture scalar anchor');source=source.replace(a,b);}
const filename=path.resolve('scripts/r4_k2_custody_fixture.js'),m=new Module(filename,module);m.filename=filename;m.paths=module.paths;m._compile(source,filename);
