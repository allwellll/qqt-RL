'use strict';
(function(root, factory) {
  const node=typeof module === 'object' && module.exports;
  const api = factory(node ? require('./sim') : root.QQT,node ? require('./review_dagger') : root.QQTReviewDagger,node ? require('./review_advantage') : root.QQTReviewAdvantage,node ? require('./review_route') : root.QQTReviewRoute,node ? require('./tactical_arena') : root.QQTTacticalArena);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTReview = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function(Q,D,A,R,T) {
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
  function validate(d) {
    const fail = message => { throw new Error(message); };
    if (!d || d.schema !== SCHEMA) fail('schema');
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
    R.validate(d);
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
