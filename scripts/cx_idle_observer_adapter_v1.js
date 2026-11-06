'use strict';
// Expose the frozen offline observer in a private module, without editing its archived source.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const src=fs.readFileSync(path.join(__dirname,'cx_idle_replay_v1.js'),'utf8');
const m=new Module(path.join(__dirname,'cx_idle_observer_private_v1.js'),module);m.filename=path.join(__dirname,'cx_idle_observer_private_v1.js');m.paths=module.paths;
m._compile(src+'\nmodule.exports.install=install;module.exports.replay=replay;\n',m.filename);
module.exports=m.exports;
