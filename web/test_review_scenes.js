'use strict';
const assert=require('assert'),fs=require('fs'),P=require('./review_protocol'),A=require('./review_advantage');
(async()=>{
  const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json'));
  const matrix=JSON.parse(fs.readFileSync('reports/replay_a1_matrix_20261017.json'));
  const level=require('../scripts/eval_web_hunter').loadLevel(),scenes=new Map();
  assert.deepEqual(catalog.scenes.map(s=>s.id),['native','open','carry_home']);
  assert.equal(catalog.replays.length,6);
  for(const declared of matrix.scenes){
    const rows=catalog.replays.filter(r=>r.scene===declared.id);
    assert.equal(rows.length,2);assert.deepEqual(rows.map(r=>r.seed).sort(),declared.seeds);
    for(const row of rows){
      const d=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
      const invalid=structuredClone(d);invalid.meta.scene='invented-map';assert.throws(()=>P.validate(invalid),/scene/);
      assert.equal(d.meta.map_sha256,await P.hash(level));
      const currentProtocol=await P.sha256(fs.readFileSync('web/review_protocol.js'));
      const legacyProtocol='77cce1b4d517a30d449348f21cccad06439fbe085d36e93a5032d3eaa62d91aa';
      assert([currentProtocol,legacyProtocol].includes(d.meta.scene_protocol_sha256),'audited source version');
      // Optional diagnostic extension changes the file hash, not scene initialization.
      if(d.meta.scene_protocol_sha256===legacyProtocol)assert.equal(await P.sha256(new TextEncoder().encode(P.setup.toString())),
        'bd6f6777a54ee218405d2721dba99cd7f73de8c05b18e8f13c04734bdddaa49d','legacy setup must remain byte identical');
      for(const [file,hash]of Object.entries(d.meta.source_hashes))assert.equal(hash,await P.sha256(fs.readFileSync(file)),'production source '+file);
      assert.equal(d.meta.reward_hashes['scripts/diagnostic_reward.py'],await P.sha256(fs.readFileSync('scripts/diagnostic_reward.py')));
      assert.match(d.meta.scene_source_sha256,/^[a-f0-9]{64}$/);
      assert.equal(d.meta.scene_sha256,await P.hash({base_map_sha256:d.meta.map_sha256,scene:d.meta.scene,max_ticks:d.meta.max_ticks,protocol_sha256:d.meta.scene_protocol_sha256}));
      const sim=P.setup(d.meta,level);scenes.set(declared.id,sim);
      if(declared.id==='native')assert(sim.brick.some(Boolean));
      else assert(sim.brick.every(n=>n===0));
      if(declared.id==='carry_home'){
        assert.equal(sim.bunCarried[0],1);assert.equal(sim.bunStored[1][1],0);
        assert.deepEqual(sim.hp,[1,1]);assert.deepEqual(Array.from(sim.pos.slice(0,2)),[9.5,4.5+d.meta.seed%3]);assert.equal(sim.maxSteps,300);
      }else{assert.equal(sim.bunCarried[0],-1);assert.equal(sim.maxSteps,2400);}
      const before=JSON.stringify(P.state(sim));
      for(const action of d.actions)sim.step(P.unpack(action));
      assert(sim.done);assert.equal(await P.hash(P.state(sim)),d.checks.at(-1)[1]);assert.notEqual(JSON.stringify(P.state(sim)),before);
      for(const mutate of [x=>x.advantages.legal_count.pop(),x=>x.advantages.legal_count[0]=0,x=>x.advantages.legal_count[0]=16,x=>x.meta.advantage.measured_count++,x=>x.meta.advantage.missing_count++,x=>x.meta.advantage.reward_hashes={},x=>x.meta.advantage.reward_hashes.extra='a'.repeat(64)]){
        const bad=structuredClone(d);mutate(bad);assert.throws(()=>A.validate(bad));
      }
    }
  }
  const html=fs.readFileSync('web/review.html','utf8');
  assert(html.indexOf('class="board"')<html.indexOf('class="playback-region"'));
  assert(html.indexOf('class="playback-region"')<html.indexOf('class="timeline-layout"'));
  assert(html.indexOf('class="timeline-layout"')<html.indexOf('class="dagger-episode"'));
  assert(/id="advantage-overlay"[^>]*checked/.test(html));
  console.log('Scenes GREEN: frozen six episodes, three genuine start/rule configurations on verified map, scene/source hashes, production Sim terminal parity, legal counts/evidence RED, playback DOM order');
})().catch(e=>{console.error(e);process.exitCode=1;});
