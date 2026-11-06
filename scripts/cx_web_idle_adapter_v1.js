'use strict';
// Private replay adapter adds browser item-slot evidence; frozen strong/review files stay intact.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
function compile(name,source){const m=new Module(path.join(__dirname,name),module);m.filename=path.join(__dirname,name);m.paths=module.paths;m._compile(source,m.filename);return m.exports;}
const original="const act = row => [S.engineAction(row[0], row[1]), S.engineAction(row[2], row[3])];";
let f=fs.readFileSync(path.join(__dirname,'cc_strong_metrics_fountain_v1.js'),'utf8');if(!f.includes(original))throw Error('replay action ABI');
f=f.replace(original,"const act = row => [0,1].map(p=>{const a=S.engineAction(row[2*p],row[2*p+1]);if(row.length>=6)a[6]=row[4+p];return a;});");
globalThis.__cxWebFountain=compile('cx_web_fountain_private_v1.js',f);
let d=fs.readFileSync(path.join(__dirname,'cx_idle_replay_v1.js'),'utf8');
d=d.replace("F=require('./cc_strong_metrics_fountain_v1')",'F=globalThis.__cxWebFountain');
const result=compile('cx_web_idle_private_v1.js',d+'\nmodule.exports.install=install;module.exports.replay=replay;\n');delete globalThis.__cxWebFountain;
module.exports=result;
