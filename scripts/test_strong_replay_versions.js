'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
if(!process.argv.includes('--child')){
 const {execFileSync}=require('node:child_process');
 for(const revision of ['current','legacy'])process.stdout.write(execFileSync(process.execPath,[__filename,'--child'],{env:{...process.env,QQT_EVAL_SIMULATOR:revision},encoding:'utf8'}));
 return;
}
const Q=require('./native_eval_simulator_v1'),E=require('./resume_gpu67_eval_v1');
const D=require('./cx_idle_observer_adapter_v1');
const hash=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const game={seed:2026110601,seat:0,mode:'greedy',opponent_mode:'greedy',spawn:[[1,1],[11,13]],native_respawn:[[1,1],[11,13]]};
const s=E.setup(game,level),checks=[[0,hash(s.snapshotReplay())]],actions=[];
while(!s.done){actions.push([4,0,4,0]);s.step([[4,0,0],[4,0,0]]);if(s.t%200===0||s.done)checks.push([s.t,hash(s.snapshotReplay())]);}
const review={meta:{learner_seat:0,model:'fixture',opponent:'fixture',game,seed:game.seed,simulator_sha256:Q.SIMULATOR_SHA256},level,actions,checks,summary:{ticks:s.t,done:s.done,winner:s.winner}};
const os=require('node:os'),path=require('node:path');
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'qqt-strong-replay-'));
try{
fs.writeFileSync(dir+'/review.json',JSON.stringify(review));D.install();const r=D.replay(dir+'/review.json');
assert.equal(r.hash_checks_verified,checks.length);assert.equal(r.ticks,2400);assert.equal(r.idle[0].ticks,2400);
review.meta.simulator_sha256='0'.repeat(64);fs.writeFileSync(dir+'/tampered.json',JSON.stringify(review));assert.throws(()=>D.replay(dir+'/tampered.json'),/simulator sha/);
console.log('PASS',Q.SIMULATOR_FILE,2400,'native ticks, observer hash checks, mismatched SHA rejected');

}finally{fs.rmSync(dir,{recursive:true,force:true});}
