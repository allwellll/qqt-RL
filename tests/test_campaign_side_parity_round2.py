import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_bomb import je07_bun_env as env
from jax_bomb import je05_core_env as base


ROOT = Path(__file__).resolve().parents[1]


def _configured_state():
    env.prepare()
    env.configure_training(
        curriculum="full=1", hit_points=5, reward_profile="legacy",
        danger_escape_reward=0.0, avoidable_danger_death_penalty=0.0,
        tactical_bomb_placement_reward=0.0,
        tactical_bomb_resolution_reward=0.0, base_bomb_reward=0.0,
        forced_kill_reward=0.0, enemy_threat_reward=0.0,
        retention_replay=0.0)
    env._DANGER_TRACKING = False
    states = env.init_batch(jax.random.PRNGKey(8801), 1,
                            jax.nn.one_hot(0, len(env.LESSON_NAMES)))
    return jax.tree.map(lambda x: x[0], states)


def _js_oracle(payload):
    js = r"""
const Q=require('./web/sim.js');
const levels=require('./web/assets/maps/levels.json');
const level=levels.find(x=>x.qqt_id===806);
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const sim=new Q.Sim(17); sim.reset(level,{nativeTrap:true,nativeItems:true});
if(input.mode==='speed'){
  const values=[sim.spdG[0]];
  for(let i=0;i<7;i++){sim._grow(0,false,2);values.push(sim.spdG[0]);}
  process.stdout.write(JSON.stringify(values));
}else if(input.mode==='contact'){
  const results=[];
  for(const sample of input.samples){
    sim.pos[0]=sample.pos[0];sim.pos[1]=sample.pos[1];
    const covered=new Uint8Array(Q.H*Q.W);
    for(const [r,c] of sample.cells)covered[r*Q.W+c]=1;
    results.push(sim._isHitByExplosion(0,covered,null,null,false));
  }
  process.stdout.write(JSON.stringify(results));
}else if(input.mode==='slow'||input.mode==='death'){
  const cell=4*Q.W+4;
  sim.wall[cell]=0;sim.brick[cell]=0;
  sim.pos.set([4.5,4.5,10.5,10.5]);
  if(input.mode==='slow'){
    sim.invuln.fill(0);sim.spawnProtection.fill(0);
    sim.fieldItem[cell]=2;sim.fieldOwner[cell]=1;sim.fieldArmed[cell]=1;
  }else{
    sim.trapped[0]=1;sim.invuln[0]=30;
  }
  sim.step([[4,0,0,0],[4,0,0,0]]);
  process.stdout.write(JSON.stringify(input.mode==='slow'
    ?[sim.movementStatus[0],sim.movementStatusTicks[0]]
    :[sim.alive[0],sim.invuln[0]]));
}else throw Error('unknown mode');
"""
    proc = subprocess.run(["node", "-e", js], cwd=ROOT, check=True,
                          text=True, input=json.dumps(payload), capture_output=True)
    return json.loads(proc.stdout)


def test_each_speed_pickup_matches_production_js():
    state = _configured_state()
    row, col = map(int, np.asarray(state.core.pos[0]).astype(np.int32))
    jax_speeds = [float(state.core.spd_g[0])]
    actions = jnp.array([[4, 0], [4, 0]], jnp.int32)
    for tick in range(7):
        crate = jnp.zeros_like(state.core.crate).at[row, col].set(3)
        state = state._replace(core=state.core._replace(crate=crate))
        state, _, _ = env.step(state, actions, jax.random.PRNGKey(9900 + tick),
                               auto_reset=False, return_info=True)
        jax_speeds.append(float(state.core.spd_g[0]))

    js_speeds = _js_oracle({"mode": "speed"})
    np.testing.assert_allclose(jax_speeds, js_speeds, atol=2e-5, rtol=0.0)
    assert jax_speeds[-1] == pytest.approx(2.4, abs=2e-5)


def test_native_trap_requires_all_current_contact_cells():
    # Player center at (5.95, 6.50) contacts cells (5,6) and (6,6).
    pos = jnp.array([[5.95, 6.50], [3.50, 13.50]], jnp.float32)
    covered = jnp.zeros((base.H, base.W), jnp.bool_)
    covered = covered.at[5, 6].set(True)
    assert bool(base._native_trap_contact_covered(pos, covered)[0]) is False
    js_results = _js_oracle({"mode": "contact", "samples": [
        {"pos": [5.95, 6.50], "cells": [[5, 6]]},
        {"pos": [5.95, 6.50], "cells": [[5, 6], [6, 6]]}]})
    assert js_results == [False, True]
    covered = covered.at[6, 6].set(True)
    assert bool(base._native_trap_contact_covered(pos, covered)[0]) is True

    covered = jnp.zeros((base.H, base.W), jnp.bool_).at[5, 6].set(True)
    assert bool(base._native_trap_contact_covered(pos, covered)[0]) is False


def test_native_slow_glue_tick_count_matches_js_phase_order():
    state = _configured_state()
    row, col = map(int, np.asarray(state.core.pos[0]).astype(np.int32))
    field_item = jnp.zeros_like(state.field_item).at[row, col].set(env.ITEM_SLOW_GLUE)
    field_owner = jnp.full_like(state.field_owner, -1).at[row, col].set(1)
    field_armed = jnp.zeros_like(state.field_armed).at[row, col].set(True)
    state = state._replace(field_item=field_item, field_owner=field_owner,
                           field_armed=field_armed)
    result, _, _ = env.step(state, jnp.array([[4, 0], [4, 0]], jnp.int32),
                            jax.random.PRNGKey(11001), auto_reset=False,
                            return_info=True)
    assert int(result.move_status[0]) == env.STATUS_SLOW
    assert int(result.status_ticks[0]) == 99
    assert _js_oracle({"mode": "slow"}) == [1, 99]


def test_native_trap_death_clears_invulnerability():
    state = _configured_state()
    state = state._replace(
        trapped=jnp.array([1, 0], jnp.int16),
        core=state.core._replace(invuln=jnp.array([30, 0], jnp.int32)))
    result, _, info = env.step(state, jnp.array([[4, 0], [4, 0]], jnp.int32),
                              jax.random.PRNGKey(12001), auto_reset=False,
                              return_info=True)
    assert bool(info["death"][0])
    assert not bool(result.core.alive[0])
    assert int(result.core.invuln[0]) == 0
    assert _js_oracle({"mode": "death"}) == [False, 0]
