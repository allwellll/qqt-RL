'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFile, execFileSync } = require('child_process');
const head=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
const out=path.resolve(process.env.EVIDENCE_FILE || 'runs/qqt_main_20261006/pages-final.json');
const base='https://allwellll.github.io/qqt-RL/';
function curl(args,input){return new Promise((resolve,reject)=>{
 const child=execFile('curl',['--silent','--show-error','--fail','--retry','2','--max-time','45',...args],{encoding:null,maxBuffer:10000000},(error,stdout)=>error?reject(new Error('HTTP resource verification failed')):resolve(stdout));
 if(input)child.stdin.end(input);
});}
(async()=>{
 let credential;
 try{credential=execFileSync('git',['credential','fill'],{input:'url=https://github.com/allwellll/qqt-RL.git\n\n',encoding:'utf8',stdio:['pipe','pipe','pipe'],env:{...process.env,GIT_TERMINAL_PROMPT:'0'}});}catch(_){throw new Error('GitHub read credential unavailable');}
 const token=credential.split('\n').find(x=>x.startsWith('password='))?.slice(9);
 if(!token)throw new Error('GitHub read credential unavailable');
 const api=async p=>JSON.parse(await curl(['--config','-'],`url = "https://api.github.com/repos/allwellll/qqt-RL/${p}"\nheader = "Authorization: Bearer ${token}"\nheader = "Accept: application/vnd.github+json"\n`));
 const [branch,runs]=await Promise.all([api('branches/main'),api('actions/workflows/pages.yml/runs?branch=main&per_page=10')]);
 if(branch.commit.sha!==head)throw new Error('Remote main differs from local HEAD');
 const run=runs.workflow_runs.find(r=>r.head_sha===head && r.event==='push');
 const record={head,remoteMain:branch.commit.sha,workflow:run?{id:run.id,status:run.status,conclusion:run.conclusion,url:run.html_url,event:run.event}:null,files:[]};
 fs.mkdirSync(path.dirname(out),{recursive:true});
 if(!run || run.status!=='completed' || run.conclusion!=='success'){
  fs.writeFileSync(out,JSON.stringify(record,null,2)+'\n');console.log(JSON.stringify(record,null,2));process.exitCode=2;return;
 }
 const info=JSON.parse(await curl([`${base}build-info.json?verify=${head}`]));
 if(info.commit!==head)throw new Error('Published build-info is not exact HEAD');record.buildInfo=info;
 const files=['index.html','app.js','style.css','leaderboard.js','leaderboard_config.js','sound.js','sim.js','controls.js','ui_panels.js','visual_renderer.js',
 'model_catalog.js','model_loader.js','replay.js','bun_coop_hunter_bot.js','bun_hunter_bot.js','assets/native/sprites.json','assets/native/maomao_portrait.png','assets/snd/吃道具音效.wav'];
 // Fetch in small batches to keep network resource usage bounded.
 for(let i=0;i<files.length;i+=4){record.files.push(...await Promise.all(files.slice(i,i+4).map(async file=>{
   const data=await curl([`${base}${file.split('/').map(encodeURIComponent).join('/')}?verify=${head}`]);
   const local=fs.readFileSync(path.join('web',file));if(!data.equals(local))throw new Error(`Published bytes differ: ${file}`);
   return {file,bytes:data.length,sha256:crypto.createHash('sha256').update(data).digest('hex')};
 })));}
 record.exactHead=true;fs.writeFileSync(out,JSON.stringify(record,null,2)+'\n');
 console.log(JSON.stringify({head,workflow:record.workflow,exactHead:true,files:record.files.length},null,2));
})().catch(error=>{console.error(error.message);process.exitCode=1;});
