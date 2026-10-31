'use strict';
const fs=require('node:fs');
const tau=Number(process.env.FRESH_TAU);
if(![0,.5].includes(tau))throw Error('frozen tau required');
if(tau===.5){
 const inference=require('./je05_web_hunter'),load=inference.loadModel;
 inference.loadModel=async(...args)=>{const model=await load(...args);require('./bomb_threat_iter2_actions').install(model.model,{tau,record:false});return model;};
}
process.on('exit',code=>{
 if(code===0){const file=process.argv[4]+'.sampling.json';fs.writeFileSync(file+'.tmp',JSON.stringify({tau,gate_input:tau===0,actor_rng:'mulberry32(seed^0x515154)',opponent_rng:'mulberry32(seed^0x7a11c0de)'}));fs.renameSync(file+'.tmp',file);}
});
require('./bomb_threat_fresh256_eval');
