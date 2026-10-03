import json
import os
os.environ.setdefault('JAXBOMB_RULE','bun')
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts import cx_trap_route_bot as bot
from scripts import cx_train_trap_routes as entry
from scripts import launch_r1_ab as launch


def test_candidate_order_and_blockers():
    free=jnp.ones((bot.H,bot.W),jnp.bool_)
    c,v=bot.trap_candidates(jnp.array([5,5]),free)
    assert c.shape==(16,2) and bool(v.all())
    np.testing.assert_array_equal(c[0],[5,5])
    blocked=free.at[5,5].set(False).at[4,5].set(False)
    c,v=bot.trap_candidates(jnp.array([5,5]),blocked)
    chosen = np.asarray(c)[np.asarray(v)]
    assert not np.any(np.all(chosen == [5,5],axis=1))
    assert not np.any(np.all(chosen == [4,5],axis=1))
    edge,valid=bot.trap_candidates(jnp.array([0,0]),free)
    assert int(valid.sum())==10
    assert np.all(np.asarray(edge)>=0)


def test_weighted_field_preserves_original_binary_goals():
    seed=jnp.zeros((bot.H,bot.W),jnp.bool_).at[5,5].set(True)
    cost=jnp.ones((bot.H,bot.W),jnp.float32).at[4,5].set(bot.BIG)
    original=bot.goal_distance(seed,cost)
    observed=bot.weighted_goal_distance(jnp.where(seed,0.,bot.BIG),cost)
    np.testing.assert_array_equal(original,observed)
    extra=bot.weighted_goal_distance(jnp.where(seed,0.,bot.BIG).at[1,1].set(2.),cost)
    assert float(extra[1,1])==2. and float(original[1,1])>2.


def test_original_goal_actions_remain_legal_and_trap_debug_is_finite():
    bot.env.prepare(); bot.env.configure_training('danger_arena=1',1)
    state=bot.env.init_batch(jax.random.PRNGKey(701),1)
    state=jax.tree.map(lambda x:x[0],state)
    state=state._replace(core=state.core._replace(brick=jnp.zeros_like(state.core.brick)))
    mm,bm=bot.env.legal_mask(state); tier=bot.B.TIER_NAMES.index('hunter_hard'); prm=jax.tree.map(lambda x:x[tier],bot.B.tier_table()); key=jax.random.PRNGKey(702)
    action,debug=jax.jit(lambda s: bot.trap_bot_action(s,1,mm[1],bm[1],key,prm,debug=True))(state)
    assert bool(mm[1,action[0]]) and bool(bm[1,action[1]])
    assert debug['trap_cells'].shape==(16,2)
    assert np.isfinite(np.asarray(debug['goal_dist'])).all()
    _,base=jax.jit(lambda s: bot.B.bot_action(s,1,mm[1],bm[1],key,prm,debug=True))(state)
    for name in ('surv','count','t_safe','deadlines','foe_count'):
        np.testing.assert_array_equal(debug[name],base[name])
    dead=state._replace(core=state.core._replace(alive=state.core.alive.at[0].set(False)))
    a=bot.B.bot_action(dead,1,mm[1],bm[1],key,prm)
    b,dbg=bot.trap_bot_action(dead,1,mm[1],bm[1],key,prm,debug=True)
    np.testing.assert_array_equal(a,b)
    assert not bool(dbg['trap_valid'].any())


def test_current_factory_and_mode_validation(monkeypatch):
    sentinel=object(); monkeypatch.setattr(entry,'HARD_FACTORY',lambda original,fraction:sentinel)
    assert entry.make_actions(None,.5,'current') is sentinel
    with pytest.raises(ValueError): entry.make_actions(None,.25,'trap16')
    with pytest.raises(ValueError): entry.make_actions(None,.5,'bad')


def test_launcher_registration_and_confirmation_seed():
    exp='EXP-CX-20261003-29'; assert len(launch.plan(exp))==8
    assert [a for a,_,_ in launch.plan(exp)]==['control']*4+['trap16']*4
    assert launch.train_argv('PY','trap16',20264001,'INIT','RUN',3,1,exp)[2]=='scripts.cx_train_trap_routes'


from collections import namedtuple

def test_hook_keeps_current_factory_and_selfplay_masks_rng(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(entry, 'HARD_FACTORY', lambda original, fraction: sentinel)
    assert entry.make_actions(None, .5, 'current') is sentinel
    with pytest.raises(ValueError): entry.make_actions(None, .25, 'trap16')
    with pytest.raises(ValueError): entry.make_actions(None, .5, 'invalid')
    n = 8; state = namedtuple('State', 'pos')(jnp.zeros((n, 2, 2)))
    mm = jnp.arange(n * 10).reshape(n * 2, 5); bm = jnp.arange(n * 6).reshape(n * 2, 3)
    sampled = jnp.tile(jnp.array([[4, 0]]), (n, 1)); key = jax.random.PRNGKey(9); seen = []
    def original(*args): seen.append(args); return sampled
    def bot(sub, players, moves, bombs, bot_key, tiers):
        assert sub.pos.shape[0] == 4
        np.testing.assert_array_equal(moves, mm[n:n+4]); np.testing.assert_array_equal(bombs, bm[n:n+4])
        np.testing.assert_array_equal(bot_key, jax.random.fold_in(key, 701))
        np.testing.assert_array_equal(players, 1)
        return jnp.tile(jnp.array([[0, 1]]), (4, 1))
    monkeypatch.setattr(entry.trap, 'rule_bot_actions', bot)
    args = (state, None, None, (mm, bm), None, key, None, None, None, None)
    result = entry.make_actions(original, .5, 'trap16')(*args)
    assert seen[0][0] is state and seen[0][5] is key
    np.testing.assert_array_equal(result[:4], [[0, 1]] * 4)
    np.testing.assert_array_equal(result[4:], sampled[4:])


@pytest.mark.parametrize('mode',['current','trap16'])
def test_entry_keeps_projected_base_and_restores_hooks(tmp_path,monkeypatch,mode):
    save=tmp_path/'run/ckpt/final.pt'; save.parent.mkdir(parents=True)
    old_factory=entry.opponents.make_opponent_actions
    old_writer=entry.training.atomic_write_json
    old_opponent_writer=entry.opponents.atomic_write_json
    argv=['entry','--trap-route-mode',mode,'--hunter-fraction','.5','--reward-shaping-scale','.6','--save',str(save)]
    monkeypatch.setattr(sys,'argv',argv)
    def probe():
        assert sys.argv[1:3]==['--value-projection','projected']
        assert '--mode' not in sys.argv
        entry.training.atomic_write_json(save.with_suffix('.json'),{'updates_completed':3})
        entry.opponents.atomic_write_json(save.parent.parent/'opponents_config.json',{})
        raise RuntimeError('hook test')
    monkeypatch.setattr(entry.projected,'main',probe)
    with pytest.raises(RuntimeError,match='hook test'): entry.main()
    assert entry.opponents.make_opponent_actions is old_factory
    assert entry.training.atomic_write_json is old_writer
    assert entry.opponents.atomic_write_json is old_opponent_writer
    assert sys.argv is argv
    assert json.loads(save.with_suffix('.json').read_text())['trap_route_mode']==mode
    metadata=json.loads((save.parent.parent/'trap_routes_config.json').read_text())
    assert metadata['value_projection']=='projected' and metadata['surviving_kill_reward']==24 and metadata['ema_decay']==.95
    from scripts import cx_confirm_round as confirm
    assert confirm.confirm_seed('EXP-CX-20261003-29',None,512)==20264011
