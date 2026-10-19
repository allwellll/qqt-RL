'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),zlib=require('zlib'),assert=require('assert');
const {execFile,execFileSync}=require('child_process');
const base='https://allwellll.github.io/qqt-RL/',repo='allwellll/qqt-RL';
const head=process.env.EXPECTED_SHA||execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
const out=process.env.EVIDENCE_FILE||'runs/replay_review_20261017/release.json';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
let token;
try{const credential=execFileSync('git',['credential','fill'],{input:`url=https://github.com/${repo}.git\n\n`,encoding:'utf8',stdio:['pipe','pipe','pipe'],env:{...process.env,GIT_TERMINAL_PROMPT:'0'}});token=credential.split('\n').find(x=>x.startsWith('password='))?.slice(9);}catch(_){}
function curl(url,auth=false){return new Promise((resolve,reject)=>{const config=`url = "${url}"\n${auth&&token?`header = "Authorization: Bearer ${token}"\n`:''}`;const p=execFile('curl',['--silent','--show-error','--fail','--retry','2','--max-time','60','--config','-'],{encoding:null,maxBuffer:30000000},(e,stdout)=>e?reject(new Error('HTTP verification failed')):resolve(stdout));p.stdin.end(config);});}
const api=async suffix=>JSON.parse(await curl(`https://api.github.com/repos/${repo}/${suffix}`,true));
(async()=>{
 const record={schema:'qqt.diagnostic-release-verification/v1',head,url:base+'review.html',files:[]};
 fs.mkdirSync(path.dirname(out),{recursive:true});
 for(let attempt=0;attempt<50;attempt++){
  const [main,runs]=await Promise.all([api('branches/main'),api('actions/workflows/pages.yml/runs?branch=main&per_page=12')]);
  assert.equal(main.commit.sha,head,'remote main changed; do not overwrite another change');
  const run=runs.workflow_runs.find(r=>r.head_sha===head&&r.event==='push');
  if(run){record.workflow={id:run.id,head_sha:run.head_sha,status:run.status,conclusion:run.conclusion,url:run.html_url};console.log('Exact SHA Pages:',run.status,run.conclusion||'pending');if(run.status==='completed'){assert.equal(run.conclusion,'success','Pages workflow failed');break;}}
  if(attempt===49)throw Error('Pages did not finish within polling deadline');
  await new Promise(resolve=>setTimeout(resolve,15000));
 }
 const info=JSON.parse(await curl(base+'build-info.json?verify='+head));assert.equal(info.commit,head,'public Pages is not expected SHA');record.build_info=info;
 const catalog=JSON.parse(await curl(base+'review-catalog.json?verify='+head));assert.equal(catalog.schema,'qqt.diagnostic-catalog/v1');
 const dagger=JSON.parse(await curl(base+'review-dagger/catalog.json?verify='+head));assert.equal(dagger.schema,'qqt.dagger-catalog/v1');
 const daggerFiles=new Set(['review-dagger/catalog.json']);for(const row of dagger.runs){daggerFiles.add('review-dagger/'+row.file);const doc=JSON.parse(fs.readFileSync('web/review-dagger/'+row.file));for(const source of doc.sources)daggerFiles.add('review-dagger/sources/'+source.file);}
 const replayRows=[...catalog.replays,...(catalog.history_replays||[])];
 const returnFiles=['review_return.js','review-evidence/return-177/checks.json','review-evidence/return-177/desktop.png','review-evidence/return-177/mobile.png',...replayRows.filter(r=>r.diagnostic_audit).map(r=>r.diagnostic_audit.file)];
 const files=['review.html','review.js','review.css','review_protocol.js','review_presentation.js','review_advantage.js','review_route.js','review_dagger.js','device_auth.js','ui_panels.js','index.html','sim.js','visual_renderer.js','review-catalog.json','review-evidence/h8-g32-critic.json','review-evidence/a1-verification.json','review-evidence/matrix.json','review-evidence/screenshots/desktop.png','review-evidence/screenshots/mobile.png','review-sim/route-v35.js','review-evidence/route-v35-critic.json','review-evidence/route-v35-g32.json','review-evidence/route-g32-motion.json','review-evidence/route-g32-browser/checks.json','review-evidence/route-g32-browser/desktop.png','review-evidence/route-g32-browser/mobile.png','assets/review-icons/play.svg','assets/review-icons/copy.svg',...replayRows.map(r=>'review-replays/'+r.file),...daggerFiles];
 const metricEvidence=fs.readdirSync('web/review-evidence/metrics-20261018').map(f=>'review-evidence/metrics-20261018/'+f);
 const goodModelEvidence=fs.existsSync('web/review-evidence/good-models-20261019')?fs.readdirSync('web/review-evidence/good-models-20261019').map(f=>'review-evidence/good-models-20261019/'+f):[];
 const metricFiles=['review_metrics.js','review_admission.js','review-replays/h8-v36-g32-carry_home-2026101785.json','review-replays/h8-v36-g32-open-2026101782.json',...metricEvidence,...goodModelEvidence];
 for(const file of [...new Set([...files,...returnFiles,...metricFiles])]){
  const bytes=await curl(base+file+'?verify='+head),local=fs.readFileSync(path.join('web',file));assert(bytes.equals(local),`public bytes mismatch: ${file}`);
  const row={file,http:200,bytes:bytes.length,sha256:sha(bytes),gzip_bytes:zlib.gzipSync(bytes,{level:9}).length};
  const expected=replayRows.find(r=>'review-replays/'+r.file===file)||dagger.runs.find(r=>'review-dagger/'+r.file===file)||replayRows.find(r=>r.diagnostic_audit?.file===file)?.diagnostic_audit;if(expected){assert.equal(row.sha256,expected.sha256);assert.equal(row.bytes,expected.bytes);assert.equal(row.gzip_bytes,expected.gzip_bytes);}record.files.push(row);
 }
 const deployments=await api('deployments?environment=github-pages&per_page=10');const deployment=deployments.find(d=>d.sha===head);assert(deployment,'same SHA Pages deployment absent');
 const statuses=await api(`deployments/${deployment.id}/statuses`);assert.equal(statuses[0].state,'success');record.deployment={sha:deployment.sha,state:statuses[0].state,url:statuses[0].environment_url};
 record.catalog={count:catalog.replays.length,screened:catalog.screened_episodes,raw_bytes:catalog.raw_bytes,gzip_bytes:catalog.gzip_bytes};record.dagger={runs:dagger.runs.map(r=>({id:r.id,scope:r.scope,bytes:r.bytes,gzip_bytes:r.gzip_bytes,sha256:r.sha256})),verified_files:daggerFiles.size};record.exact_head=true;fs.writeFileSync(out,JSON.stringify(record,null,2)+'\n');
 console.log(JSON.stringify({head,workflow:record.workflow,deployment:record.deployment,catalog:record.catalog,verified_files:record.files.length},null,2));
})().catch(e=>{console.error(e.message);process.exitCode=1;});
