'use strict';
const fs=require('fs'),zlib=require('zlib'),assert=require('assert'),crypto=require('crypto');
const P=require('../web/review_protocol'),G=require('./generate_diagnostic_replays'),A=require('../web/review_advantage');
const stage=process.env.REPLAY_STAGE||'h7';
const source=process.env.REPLAY_EPISODES||'runs/replay_review_20261017/episodes',out='web/review-replays';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
(async()=>{
  const files=fs.readdirSync(source).filter(n=>new RegExp(`^${stage}-g32-.*\\.json$`).test(n)).sort();assert.equal(files.length,24);
  const documents=files.map(file=>JSON.parse(fs.readFileSync(`${source}/${file}`))).sort((a,b)=>a.meta.seed-b.meta.seed);
  if(stage==='h8'){
    const seeds=[...Array.from({length:12},(_,i)=>2026101740+i),...Array.from({length:12},(_,i)=>2026101760+i)];
    assert.deepEqual(documents.map(d=>d.meta.seed),seeds,'predeclared seed set');
    for(let i=0;i<24;i++){
      assert.equal(documents[i].meta.variant,i>=6&&i<12?'open':'native');
      assert.equal(documents[i].meta.opponent,i<12?'bun.hunter/normal':i<18?'bun.hunter/hard':'bun.tactical/v2');
    }
  }
  for(const d of documents){
    const sim=P.setup(d.meta,require('./eval_web_hunter').loadLevel()),track={idle:0,reverse:0,distance:G.distance(sim,0)},events=[];
    for(const a of d.actions){const before=P.state(sim),info=sim.step(P.unpack(a));events.push(...G.events(before,sim,info,track));}
    events.push(...d.events.filter(e=>e.kind==='self_bomb'&&!events.some(x=>x.kind===e.kind&&x.tick===e.tick)));
    d.events=events.sort((a,b)=>a.tick-b.tick);
    d.summary.failures=['enemy_drop_unpicked','window_closed_empty','return_deviation','cage_no_delivery','self_bomb','stagnation'].filter(k=>events.some(e=>e.player===0&&e.kind===k));
  }
  const selected=[],covered=new Set();
  for(const kind of ['enemy_drop_unpicked','window_closed_empty','return_deviation','cage_no_delivery','self_bomb','stagnation']){
    const d=documents.find(d=>d.summary.failures.includes(kind));if(d&&!selected.includes(d))selected.push(d);
  }
  for(const d of documents)if(d.summary.failures.length&&!selected.includes(d)&&selected.length<6)selected.push(d);
  assert(selected.length>0);
  fs.mkdirSync(out,{recursive:true});
  const model=await G.load(),evidence=[];
  for(const d of selected){
    assert(d.meta.reward_source && d.rewards.length,'real reward required');
    const opponent=d.meta.opponent==='bun.tactical/v2'?'tactical':d.meta.opponent.split('/')[1];
    const rerun=await G.episode(d.meta.seed,d.meta.variant,model,true,opponent);
    assert.deepEqual(rerun.actions,d.actions,'fresh real-policy actions');assert.deepEqual(rerun.checks,d.checks,'fresh terminal/state hashes');
    for(const key of ['ticks','done','winner','score','stored'])assert.deepEqual(rerun.summary[key],d.summary[key],key);
    d.meta.rules={native_trap:true,native_items:true,tick_hz:10,max_ticks:d.meta.max_ticks};
    if(stage==='h8'){
      d.meta.advantage=A.missing({...d.meta,gamma:model.proof.gamma});d.advantages=A.empty();
      d.meta.input_proof={...model.proof};
    }
    d.meta.replay_identity=await P.hash(P.identity(d));
    P.validate(d);const player=new P.Player(d,require('./eval_web_hunter').loadLevel());await player.verify();
    const bytes=Buffer.from(JSON.stringify(d)),file=`${d.meta.id}.json`;fs.writeFileSync(`${out}/${file}`,bytes);
    selected[selected.indexOf(d)]=d;d.summary.failures.forEach(k=>covered.add(k));
    evidence.push({id:d.meta.id,actions_exact:true,state_hashes_exact:true,terminal_exact:true,sha256:sha(bytes),bytes:bytes.length,gzip_bytes:zlib.gzipSync(bytes,{level:9}).length});
    console.log(d.meta.id,bytes.length,'bytes',evidence.at(-1).gzip_bytes,'gzip; policy actions exact');
  }
  await model.session.release();
  const replayRows=selected.map(d=>{const size=evidence.find(e=>e.id===d.meta.id);return {id:d.meta.id,file:`${d.meta.id}.json`,seed:d.meta.seed,variant:d.meta.variant,model:d.meta.model,opponent:d.meta.opponent,checkpoint_sha256:d.meta.checkpoint_sha256,summary:d.summary,sha256:size.sha256,bytes:size.bytes,gzip_bytes:size.gzip_bytes};});
  const catalog={schema:'qqt.diagnostic-catalog/v1',model:stage==='h8'?'H8 v34 g32 raw':'H7 v33 g32 raw',screened_episodes:documents.length,seed_ranges:stage==='h8'?[[2026101740,2026101751],[2026101760,2026101771]]:[[2026101700,2026101711],[2026101720,2026101731]],selection:'first episode per declared failure, then earliest failed episodes, maximum six',observed_failures:[...covered],unobserved_failures:['enemy_drop_unpicked','return_deviation','cage_no_delivery','self_bomb','stagnation'].filter(k=>!covered.has(k)),raw_bytes:evidence.reduce((n,e)=>n+e.bytes,0),gzip_bytes:evidence.reduce((n,e)=>n+e.gzip_bytes,0),replays:replayRows};
  if(stage==='h8'){
    const old=JSON.parse(fs.readFileSync('web/review-catalog.json'));
    catalog.history_replays=old.history_replays||old.replays;
  }
  fs.writeFileSync('web/review-catalog.json',JSON.stringify(catalog,null,2)+'\n');
  const screen={schema:'qqt.diagnostic-screen/v1',seed_ranges:catalog.seed_ranges,selection:catalog.selection,episodes:documents.map(d=>({id:d.meta.id,episode_id:d.meta.episode_id,seed:d.meta.seed,variant:d.meta.variant,opponent:d.meta.opponent,selected:selected.some(s=>s.meta.id===d.meta.id),summary:d.summary})),verification:evidence};
  fs.writeFileSync(`reports/replay_screen_${stage==='h8'?'h8_':''}20261017.json`,JSON.stringify(screen,null,2)+'\n');
  fs.writeFileSync(`runs/replay_review_20261017/${stage==='h8'?'h8/':''}inference-verification.json`,JSON.stringify({cpu_only:true,onnx_js_max_error:model.getParity(),advantage_status:stage==='h8'?'not_recorded':null,evidence},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
