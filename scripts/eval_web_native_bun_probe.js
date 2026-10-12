'use strict';
// Independent safety reference; never changes eval128 or danger-arena Hunter.
const assert = require('node:assert/strict');
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const QQT = require('../web/sim.js');
const authority = require('./eval_web_hunter.js');
const residual = require('./eval_web_hunter_residual.js');
const SEED = 392010731;
const MODES = ['native_1v1', 'rescue_2v1'];

function setup(level, seed, mode) {
  assert.ok(MODES.includes(mode));
  const teams = mode === 'rescue_2v1' ? [0, 1, 0] : [0, 1];
  const opts = {teams, nativeItems: true, nativeTrap: true};
  const sim = new QQT.Sim(seed, opts);
  sim.reset(level, opts);
  sim.maxSteps = 300;
  if (mode === 'rescue_2v1') {
    const y = Math.floor(sim.pos[0]), x = Math.floor(sim.pos[1]);
    const queue = [[y, x, 0]], seen = new Set([y*QQT.W+x]);
    let cell = null;
    while (queue.length && !cell) {
      const [r, c, distance] = queue.shift();
      if (distance === 2) {cell = [r, c]; break;}
      for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nr = r+dr, nc = c+dc, index = nr*QQT.W+nc;
        if (nr < 0 || nr >= QQT.H || nc < 0 || nc >= QQT.W || seen.has(index)
            || sim.wall[index] || sim.brick[index] || sim.pushable[index]) continue;
        seen.add(index); queue.push([nr,nc,distance+1]);
      }
    }
    assert.ok(cell, 'legal reachable rescue opportunity required');
    sim.pos[4] = cell[0]+.5; sim.pos[5] = cell[1]+.5;
    sim.bunSpawnPos[2] = [sim.pos[4],sim.pos[5]];
    sim.trapped[2] = sim.trapTicks;
  }
  return sim;
}

function aliasAudit(sim) {
  assert.equal(sim.nPlayers, 3);
  const observe = () => [Array.from(sim.encodeObsJAX(0,24)),Array.from(sim.encodeStateJAX(0))];
  const initial = observe(), before = [sim.pos[4],sim.pos[5],sim.trapped[2]];
  sim.pos[4] = sim.pos[0]; sim.pos[5] = sim.pos[1];
  const moved = observe();
  sim.pos[4] = before[0]; sim.pos[5] = before[1]; sim.trapped[2] = 0;
  const freed = observe();
  sim.trapped[2] = before[2];
  return {teammate_location_unobserved: JSON.stringify(initial) === JSON.stringify(moved),
    trap_state_unobserved: JSON.stringify(initial) === JSON.stringify(freed),
    scope: 'observation and state tensors only; action masks can still reflect self-trapping'};
}

function newHeroTrap(before, after, info, player) {
  return Number(before[player] <= 0 && after[player] > 0 &&
    (!!info.physicalDamageSource[player][0] || !!info.causalDamageSource[player][0]));
}

function runEpisode(model, difficulty, mode) {
  const ordinal = ['hard','normal','easy'].indexOf(difficulty)*2+MODES.indexOf(mode);
  const seed = (SEED+ordinal*7919) >>> 0;
  const sim = setup(authority.loadLevel(), seed, mode);
  const hunter = authority.createHunter(difficulty);
  hunter.adapter.reset({schema:'qqt.bot.context/v1', episode_id:`native-${ordinal}`, seed,
    ruleset:'bun',max_ticks:300,metadata:{}});
  const rng = QQT.mulberry32(seed ^ 0x515154);
  const spawn = Array.from(sim.pos);
  const audit = mode === 'rescue_2v1' ? aliasAudit(sim) : null;
  const metrics = {actor_deaths:0,actor_traps:0,self_caused_traps:0,friendly_caused_traps:0,
    teammate_deaths:0,rescues:0,bombs:0,stationary_ticks:0,rescue_first_tick:null,
    initial_rescue_failed:0};
  let initialOpportunity = mode === 'rescue_2v1';
  function step(settling) {
    const before = sim.trapped.slice(), pos = sim.pos.slice();
    const raw = model.act(sim,0,rng);
    const bot = hunter.adapter.act({schema:'qqt.bot.observation/v1',tick:sim.t,state:null,
      legal_moves:[0,1,2,3,4],legal_abilities:[0,1,2],metadata:{sim}},1,rng);
    const actions = [[Number(raw[0]),settling?0:Number(raw[1]),settling?0:Number(raw[2])||0],
      [bot.move,settling?0:Number(bot.ability===1),settling?0:Number(bot.ability===2)]];
    if (sim.nPlayers === 3) actions.push([4,0,0]);
    if (settling) sim.done = false;
    const info = sim.step(actions);
    metrics.actor_deaths += Number(info.died[0]);
    metrics.actor_traps += Number(before[0]<=0 && sim.trapped[0]>0);
    metrics.self_caused_traps += newHeroTrap(before,sim.trapped,info,0);
    metrics.bombs += Number(info.placed[0]);
    metrics.stationary_ticks += Number(Math.abs(pos[0]-sim.pos[0])+Math.abs(pos[1]-sim.pos[1]) < 1e-6);
    if (sim.nPlayers === 3) {
      metrics.friendly_caused_traps += newHeroTrap(before,sim.trapped,info,2);
      metrics.teammate_deaths += Number(info.died[2]);
      if (before[2]>0 && sim.trapped[2]===0 && !info.died[2] && sim.alive[2]) {
        metrics.rescues++;
        if (initialOpportunity) {metrics.rescue_first_tick=sim.t; initialOpportunity=false;}
      } else if (initialOpportunity && info.died[2]) {
        metrics.initial_rescue_failed=1; initialOpportunity=false;
      }
    }
    if (settling) assert.ok(!info.placed.some(Boolean));
  }
  while (!sim.done && sim.t < 300) step(false);
  const primaryTicks = sim.t;
  sim.maxSteps = primaryTicks+125;
  while (residual.hasHazards(sim) && sim.t < primaryTicks+124) step(true);
  assert.ok(!residual.hasHazards(sim), 'native residual not settled within124ticks');
  return {ordinal,seed,difficulty,mode,spawn,primary_ticks:primaryTicks,ticks:sim.t,
    hazards_remaining:false,metrics,final_alive:sim.alive.slice(),alias_audit:audit,
    scope:'native maps/items/traps retained; rescue scenario is intentional stress test, not win-rate estimate'};
}

function runShard(options) {
  assert.equal(process.env.CUDA_VISIBLE_DEVICES || '', '');
  const loaded = authority.loadModel(options.modelPath);
  const episodes = MODES.map(mode => runEpisode(loaded.model,options.difficulty,mode));
  const root = path.resolve(__dirname,'..');
  const names = ['web/sim.js','web/bun_hunter_bot.js','web/bot_contract.js','web/assets/maps/levels.json',
    'scripts/eval_web_hunter.js','scripts/eval_web_hunter_residual.js','scripts/eval_web_native_bun_probe.js'];
  return {schema:'native_bun_safety_probe_v1',checkpoint_sha256:options.checkpointSha,
    source_hashes:Object.fromEntries(names.map(name => [name,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,name))).digest('hex')])),
    model_json_sha256:loaded.sha,seed:SEED,difficulty:options.difficulty,cpu_only:true,
    native_items:true,native_trap:true,bricks_retained:true,real_web_hunter:true,
    primary_limit:300,residual_limit:124,episodes};
}

function parseOptions(argv) {
  const args = {};
  assert.equal(argv.length % 2, 0);
  for (let i=0;i<argv.length;i+=2) {
    assert.ok(argv[i].startsWith('--'));
    args[argv[i].slice(2)] = argv[i+1];
  }
  for (const key of ['model','difficulty','seed','out']) assert.ok(args[key],`missing --${key}`);
  assert.equal(Number(args.seed),SEED,'native probe seed must be frozen');
  if (args.start) assert.equal(Number(args.start),0);
  if (args.end) assert.equal(Number(args.end),2);
  if (args['max-steps']) assert.equal(Number(args['max-steps']),300);
  return {modelPath:args.model,difficulty:args.difficulty,checkpointSha:args['checkpoint-sha'],out:args.out};
}

if (require.main === module) {
  const options = parseOptions(process.argv.slice(2));
  const result = runShard(options);
  fs.writeFileSync(`${options.out}.tmp`,JSON.stringify(result));
  fs.renameSync(`${options.out}.tmp`,options.out);
}
module.exports = {setup,aliasAudit,newHeroTrap,runEpisode,runShard,parseOptions};
