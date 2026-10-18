'use strict';
(function(root, factory) {
  const node=typeof module === 'object' && module.exports;
  const api = factory(node ? require('./sim') : root.QQT,node ? require('./review_dagger') : root.QQTReviewDagger,node ? require('./review_advantage') : root.QQTReviewAdvantage,node ? require('./review_route') : root.QQTReviewRoute,node ? require('./tactical_arena') : root.QQTTacticalArena,node ? require('./review_metrics') : root.QQTReviewMetrics);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTReview = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function(Q,D,A,R,T,M) {
  const SCHEMA = 'qqt.diagnostic-replay/v1';
  const ANNOTATIONS = 'qqt.replay-annotations/v1';
  const ROUTE_SIM_HASH='9aa8385489affdfdb61c1deff61cee3d8401e744cf80f2b474feb931103884e3';
  const versions=new Map();
  function registerSimulator(hash,api){if(hash!==ROUTE_SIM_HASH)throw Error('unknown simulator');versions.set(hash,api);}
  function simulator(meta){
    if(meta.simulator_sha256!==ROUTE_SIM_HASH)return Q;
    if(!versions.has(ROUTE_SIM_HASH)&&typeof module==='object'&&module.exports)registerSimulator(ROUTE_SIM_HASH,require('./review-sim/route-v35'));
    if(!versions.has(ROUTE_SIM_HASH))throw Error('versioned simulator not loaded');
    return versions.get(ROUTE_SIM_HASH);
  }
  function replaySetup(meta,level){
    if(meta.native_fixture){
      validateFixture(meta.native_fixture);
      const sim=new Q.Sim(meta.seed);
      sim.reset(level,{nativeTrap:true,nativeItems:true});
      sim.restoreReplay(meta.native_fixture.frame);
      sim.initialHp=meta.native_fixture.initial_hp;
      sim.maxSteps=meta.max_ticks;
      return sim;
    }

    if(meta.scene!=='route_drop_return')return meta.arena||meta.native_rules===false||meta.truncated?setupExtended(meta,level):setup(meta,level);
    const api=simulator(meta),s=new api.Sim(meta.seed);
    s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=1200;
    s.initialHp=1;s.hp=[1,1];s.invuln=[0,0];s.brick.fill(0);
    s.pos.set([5.5,5.5,5.5,9.5]);s.bunCarried[1]=1;s.bunStored[1][1]=0;s.trapped[1]=1;
    s.bunSpawnPos=[[s.pos[0],s.pos[1]],[s.pos[2],s.pos[3]]];s._gen++;return s;
  }
  const stringify = value => JSON.stringify(value);
  async function sha256(bytes) {
    const crypto = typeof module === 'object' && module.exports ? require('crypto').webcrypto : globalThis.crypto;
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
  }
  const hash = value => sha256(new TextEncoder().encode(stringify(value)));
  function validateFixture(f){
    const tasks=['native_death_stock','native_sugar_stock','real_drop','carry_near','carry_mid','carry_far','recovery_threat'];
    if(!f||f.schema!=='qqt.native-reset-fixture/v1'||!tasks.includes(f.task)||f.initial_hp!==1||
      !/^[a-f0-9]{64}$/.test(f.frame_sha256||'')||f.frame?.t!==0||f.frame.nativeTrap!==true||f.frame.nativeItems!==true||
      f.frame.done!==false||f.frame.nPlayers!==2||f.frame.pos?.length!==4||f.frame.wall?.length!==195||
      f.frame.brick?.length!==195||f.frame.bunLoose?.length!==390||
      f.frame.pos.some(v=>!Number.isFinite(v)||v<0||v>15))throw Error('native reset fixture');
  }
  function validate(d) {
    const fail = message => { throw new Error(message); };
    if (!d || d.schema !== SCHEMA) fail('schema');
    if(d.meta?.native_fixture)validateFixture(d.meta.native_fixture);
    if (!d.meta || !/^[\w-]+$/.test(d.meta.id) || !Number.isSafeInteger(d.meta.seed) || d.meta.tick_ms !== 100) fail('meta');
    if (d.meta.scene !== undefined && !['native','open','carry_home','route_drop_return'].includes(d.meta.scene)) fail('scene');
    if(d.meta.arena!==undefined&&d.meta.arena!=='tactical_v2')fail('arena');
    if(d.meta.native_rules!==undefined&&typeof d.meta.native_rules!=='boolean')fail('native rules');
    if(d.meta.arena!==undefined&&!/^[a-f0-9]{64}$/.test(d.meta.arena_map_sha256||''))fail('arena map hash');
    if(d.meta.scene==='route_drop_return'&&(d.meta.simulator_sha256!==ROUTE_SIM_HASH||d.meta.diagnostic!==true||d.meta.reward_status!=='unrecorded'||d.meta.training_reward_status!=='unrecorded'))fail('diagnostic evidence');
    for (const k of ['checkpoint_sha256','model_sha256','simulator_sha256','opponent_sha256','map_sha256','replay_identity']) if (!/^[a-f0-9]{64}$/.test(d.meta[k])) fail(k);
    if (!Array.isArray(d.actions) || !d.actions.length || d.actions.length > 10000 || d.actions.length !== d.summary?.ticks) fail('length');
    for (const a of d.actions) if (!Array.isArray(a) || a.length !== 4 || a.some((v,i) => !Number.isInteger(v) || v < 0 || v > (i%2 ? 2 : 4))) fail('illegal action');
    let previous = 0;
    const totals = [0,0];
    if (!Array.isArray(d.rewards)) fail('rewards');
    for (const row of d.rewards) {
      if (!Array.isArray(row) || row.length !== 3 || !Number.isInteger(row[0]) || row[0] <= previous || row[0] > d.actions.length) fail('reward tick');
      previous = row[0];
      for (let p=0;p<2;p++) {
        if (!row[p+1] || Array.isArray(row[p+1]) || typeof row[p+1] !== 'object') fail('reward components');
        for (const [key,value] of Object.entries(row[p+1])) {
          if (!/^[a-z][a-z0-9_]*$/.test(key) || !Number.isFinite(value) || value === 0) fail('reward delta');
          totals[p] += value;
        }
      }
    }
    if(d.meta.reward_status==='unrecorded'){
      if(d.rewards.length||JSON.stringify(d.summary.reward)!=='[null,null]')fail('unrecorded reward');
    }else if (!Array.isArray(d.summary.reward) || d.summary.reward.length !== 2 || totals.some((v,p) => !Number.isFinite(d.summary.reward[p]) || Math.abs(v-d.summary.reward[p]) > 1e-4)) fail('reward sum');
    if (!Array.isArray(d.events)) fail('events');
    for (const e of d.events) if (!Number.isInteger(e.tick) || e.tick < 0 || e.tick > d.actions.length || typeof e.kind !== 'string') fail('event');
    D.validateEvents(d.dagger_events,d.actions.length);
    A.validate(d);
    M.validate(d);
    R.validate(d);
    if(d.meta.offline_recomputed_current_reward){
      const m=d.meta.offline_recomputed_current_reward;
      if(m.status!=='offline_recomputed_current_reward'||d.meta.reward_status!==m.status||d.meta.training_reward_status!=='unrecorded'||
        !Array.isArray(m.assumptions)||!m.assumptions.length||!Array.isArray(m.component_names)||!m.component_names.length||
        !Array.isArray(d.diagnostic_ticks)||d.diagnostic_ticks.length!==d.actions.length||m.measured_count!==d.actions.length)fail('offline reward scope');
      let previousPhi=null;
      const totals=[0,0];
      for(let i=0;i<d.diagnostic_ticks.length;i++){
        const row=d.diagnostic_ticks[i],r=row.offline_recomputed_current_reward;
        if(row.tick!==i+1||['training_reward','raw_a1','shaped_a1','cross_critic_delta_a','raw_time_delta_a','shaped_time_delta_a'].some(k=>row[k]!==null)||!r||r.status!==m.status||
          r.components?.length!==2||r.approx_total_reward?.length!==2)fail('offline reward tick');
        const decision=row.decision;
        if(!decision||JSON.stringify(decision.action)!==JSON.stringify(d.actions[i].slice(0,2)))fail('offline actor action');
        for(const [probability,mask,size]of [[decision.move_probability,decision.move_mask,5],[decision.ability_probability,decision.ability_mask,3]]){
          if(probability?.length!==size||mask?.length!==size||probability.some((v,j)=>!Number.isFinite(v)||v<0||(!mask[j]&&v!==0))||
            Math.abs(probability.reduce((a,b)=>a+b,0)-1)>1e-6)fail('offline actor probability');
        }
        for(let p=0;p<2;p++){
          if(Object.keys(r.components[p]).sort().join(',')!==[...m.component_names].sort().join(',')||
            Object.values(r.components[p]).some(v=>!Number.isFinite(v))||!Number.isFinite(r.approx_total_reward[p])||
            Math.abs(Object.values(r.components[p]).reduce((a,b)=>a+b,0)-r.approx_total_reward[p])>2e-5)fail('offline reward component sum');
          totals[p]+=r.approx_total_reward[p];
        }
        const route=r.route[0];
        if(!route||!Object.values(route).every(Number.isFinite)||Math.abs(route.route_potential_delta-(.995*route.route_phi_after-route.route_phi_before))>2e-5||
          Math.abs(route.route_shaped_reward-.05*route.route_potential_delta)>2e-6||
          Math.abs(r.components[0].route_shaped_reward-route.route_shaped_reward)>2e-6||
          (previousPhi===null?r.approx_phi_jump!==null:!Number.isFinite(r.approx_phi_jump)||Math.abs(route.route_phi_before-previousPhi-r.approx_phi_jump)>1e-7))fail('offline route formula');
        previousPhi=route.route_phi_after;
        const sparse=d.rewards.find(x=>x[0]===i+1);
        for(let p=0;p<2;p++)for(const k of m.component_names)if((sparse?.[p+1]?.[k]||0)!==r.components[p][k])fail('offline sparse components');
      }
      if(totals.some((v,p)=>Math.abs(v-d.summary.reward[p])>1e-4)||previousPhi!==0)fail('offline reward terminal');
    }
    if(d.meta.approximate_a1){
      const m=d.meta.approximate_a1,rows=d.diagnostic_ticks;
      if(m.schema!=='qqt.je10-approx-a1/v1'||m.status!=='offline_approximation'||m.future_value!==0||m.measured_count!==d.actions.length||rows?.length!==d.actions.length||!m.definition||!m.limitations)fail('approximate A1 scope');
      for(let i=0;i<rows.length;i++){
        const r=rows[i],a=r.approximate_a1,cs=r.counterfactuals;
        if(r.tick!==i+1||r.raw_a1!==null||r.shaped_a1!==null||r.training_reward!==null||!a||!cs?.length||cs.length>15)fail('approximate A1 tick');
        const mass=cs.reduce((n,c)=>n+c.probability,0),taken=cs.find(c=>c.action[0]===d.actions[i][0]&&c.action[1]===d.actions[i][1]);
        if(!taken||Math.abs(mass-1)>1e-6||cs.some(c=>!Number.isFinite(c.probability)||c.probability<0||!Number.isFinite(c.raw_reward)||!Number.isFinite(c.shaped_reward)))fail('approximate A1 candidates');
        const unique=new Set(cs.map(c=>c.action.join(',')));
        if(unique.size!==cs.length||cs.length!==r.decision.move_mask.filter(Boolean).length*r.decision.ability_mask.filter(Boolean).length||cs.some(c=>!r.decision.move_mask[c.action[0]]||!r.decision.ability_mask[c.action[1]]||Math.abs(c.probability-r.decision.move_probability[c.action[0]]*r.decision.ability_probability[c.action[1]])>1e-8))fail('approximate A1 legal policy');
        for(const role of ['raw','shaped']){
          const mean=cs.reduce((n,c)=>n+c.probability*c[role+'_reward'],0)/mass;
          if(![a[role+'_a1'],a[role+'_q1_taken'],a[role+'_q1_mean']].every(Number.isFinite)||Math.abs(a[role+'_q1_taken']-taken[role+'_reward'])>1e-10||Math.abs(a[role+'_q1_mean']-mean)>1e-10||Math.abs(a[role+'_a1']-(taken[role+'_reward']-mean))>1e-10)fail('approximate A1 centering');
        }
        if(!Number.isFinite(a.delta_a)||Math.abs(a.delta_a-(a.shaped_a1-a.raw_a1))>1e-10)fail('approximate A1 gap');
        for(const k of ['raw_a1','shaped_a1','delta_a'])if(i?(!Number.isFinite(a[k+'_increment'])||Math.abs(a[k+'_increment']-(a[k]-rows[i-1].approximate_a1[k]))>1e-10):a[k+'_increment']!==null)fail('approximate A1 increment');
      }
    }
    previous = -1;
    for (const row of d.checks || []) {
      if (!Array.isArray(row) || row.length !== 2 || !Number.isInteger(row[0]) || row[0] <= previous || row[0] > d.actions.length || !/^[a-f0-9]{64}$/.test(row[1])) fail('state hash');
      previous = row[0];
    }
    const truncated=d.summary.done===false&&d.meta.truncated===true&&d.actions.length===d.meta.max_ticks;
    if (d.checks?.[0]?.[0] !== 0 || previous !== d.actions.length || (d.summary.done!==true&&!truncated)) fail('terminal');
    return d;
  }
  function setup(meta, level) {
    const sim = new Q.Sim(meta.seed);
    const scene = meta.scene || meta.variant || 'native';
    sim.reset(level, {nativeTrap:true,nativeItems:true});
    if (scene === 'open' || meta.variant === 'open') sim.brick.fill(0);
    if (scene === 'carry_home') {
      // je05_web_tasks.setup('carry_home'): declared initial task state only.
      sim.brick.fill(0);
      sim.initialHp=1;sim.hp=[1,1];sim.invuln=[0,0];
      sim.pos[0]=9.5;sim.pos[1]=4.5+(meta.seed%3);
      sim.bunCarried[0]=1;sim.bunStored[1][1]=0;
      sim.bunSpawnPos[0]=[sim.pos[0],sim.pos[1]];
    }
    sim.maxSteps = meta.max_ticks;
    return sim;
  }
  function setupExtended(meta,level){
    if(meta.arena==='tactical_v2'){
      if(!T)throw Error('tactical arena support unavailable');
      level=T.buildTacticalArena(level);
    }
    const sim=new Q.Sim(meta.seed),scene=meta.scene||meta.variant||'native';
    sim.reset(level,meta.native_rules===false?{}:{nativeTrap:true,nativeItems:true});
    if(scene==='open'||meta.variant==='open')sim.brick.fill(0);
    if(scene==='carry_home'){
      sim.brick.fill(0);sim.initialHp=1;sim.hp=[1,1];sim.invuln=[0,0];
      sim.pos[0]=9.5;sim.pos[1]=4.5+(meta.seed%3);sim.bunCarried[0]=1;sim.bunStored[1][1]=0;
      sim.bunSpawnPos[0]=[sim.pos[0],sim.pos[1]];
    }
    sim.maxSteps=meta.truncated?Math.max(meta.max_ticks+1,2400):meta.max_ticks;
    return sim;
  }
  const unpack = row => [[row[0],Number(row[1]===1),Number(row[1]===2)],[row[2],Number(row[3]===1),Number(row[3]===2)]];
  function state(sim) { return sim.snapshotReplay(); }
  class Player {
    constructor(d, level) { this.document = validate(d); this.level = level; this.cache = new Map(); this.reset(); }
    countRandom(sim, rng, draws = 0) {
      this.draws = draws;
      sim.rng = () => { this.draws++; return rng(); };
    }
    checkpoint() {
      const data = Object.fromEntries(Object.entries(this.sim).filter(([,v]) => typeof v !== 'function'));
      this.cache.set(this.sim.t, {data:structuredClone(data),draws:this.draws,info:structuredClone(this.info)});
    }
    reset() {
      this.sim = replaySetup(this.document.meta,this.level); this.info = null;
      this.countRandom(this.sim,this.sim.rng);
      if (!this.cache.has(0)) this.checkpoint();
    }
    restore(tick) {
      const cached = this.cache.get(tick), fresh = replaySetup(this.document.meta,this.level);
      // Rebuild the exact seeded random stream; snapshots alone omit its closure.
      for (let i=0;i<cached.draws;i++) fresh.rng();
      this.sim = Object.assign(Object.create(Object.getPrototypeOf(fresh)),structuredClone(cached.data));
      this.info = structuredClone(cached.info);
      this.countRandom(this.sim,fresh.rng,cached.draws);
    }
    step() {
      if (this.sim.t >= this.document.actions.length) return false;
      this.info = this.sim.step(unpack(this.document.actions[this.sim.t]));
      if (this.sim.t%200 === 0) this.checkpoint();
      return true;
    }
    seek(tick) {
      if (!Number.isInteger(tick) || tick < 0 || tick > this.document.actions.length) throw new Error('seek range');
      const nearest = Math.max(...[...this.cache.keys()].filter(t => t<=tick));
      if (tick < this.sim.t || nearest > this.sim.t) this.restore(nearest);
      while (this.sim.t < tick) this.step();
      return this.sim;
    }
    async verify() {
      const fixture=this.document.meta.native_fixture;
      if(fixture&&await hash(fixture.frame)!==fixture.frame_sha256)throw Error('native fixture frame hash');
      if (await hash(identity(this.document)) !== this.document.meta.replay_identity) throw new Error('replay identity hash');
      this.cache.clear();
      this.reset();
      for (const [tick,expected] of this.document.checks) {
        this.seek(tick);
        if (await hash(state(this.sim)) !== expected) throw new Error(`state hash @${tick}`);
      }
      if (this.sim.done !== this.document.summary.done || this.sim.winner !== this.document.summary.winner) throw new Error('terminal mismatch');
      this.reset(); return true;
    }
  }
  function identity(d){
    const value={meta:{...d.meta,replay_identity:undefined},actions:d.actions};
    if(d.dagger_events!==undefined)value.dagger_events=d.dagger_events;
    if(d.advantages!==undefined)value.advantages=d.advantages;
    if(d.route!==undefined)value.route=d.route;
    if(d.diagnostic_ticks!==undefined)value.diagnostic_ticks=d.diagnostic_ticks;
    if(d.metrics!==undefined)value.metrics=d.metrics;
    return value;
  }
  function time(tick, ms=100) {
    const total = tick*ms;
    return `${String(Math.floor(total/60000)).padStart(2,'0')}:${String(Math.floor(total/1000)%60).padStart(2,'0')}.${String(total%1000).padStart(3,'0')}`;
  }
  const replayHash = d => d.asset_sha256 || d.meta.replay_identity;
  const locator = (d,tick) => `${d.meta.id}@tick=${tick} (${time(tick,d.meta.tick_ms)}) sha256=${replayHash(d)}`;
  function annotationDocument(d, rows) { return {schema:ANNOTATIONS,replay_id:d.meta.id,replay_hash:replayHash(d),annotations:rows}; }
  function validateAnnotations(value,d) {
    if (value?.schema !== ANNOTATIONS || value.replay_id !== d.meta.id || value.replay_hash !== replayHash(d) || !Array.isArray(value.annotations) || value.annotations.length > 2000) throw new Error('标注 schema / replay ID / hash 不匹配');
    const ids = new Set();
    for (const a of value.annotations) {
      if (typeof a.id !== 'string' || !/^[\w-]{1,80}$/.test(a.id) || ids.has(a.id) || !Number.isInteger(a.tick) || a.tick<0 || a.tick>d.actions.length || !['pickup','window','return','delivery','safety','other'].includes(a.category) || !['low','medium','high'].includes(a.severity) || typeof a.note !== 'string' || a.note.length>2000) throw new Error('非法标注');
      ids.add(a.id);
    }
    return value.annotations;
  }
  function phase(sim,p=0) {
    if (!sim.alive[p]) return '死亡 / 复活';
    if (sim.trapped[p]>0) return '糖泡控制';
    if (sim.bunCarried[p]>=0) return '持包返程';
    if (!sim.alive[1-p] || sim.trapped[1-p]>0) return '运输窗口';
    return '战斗';
  }
  return {SCHEMA,ANNOTATIONS,ROUTE_SIM_HASH,registerSimulator,simulator,replaySetup,validate,identity,hash,sha256,stringify,setup,unpack,state,Player,time,locator,annotationDocument,validateAnnotations,phase};
});
