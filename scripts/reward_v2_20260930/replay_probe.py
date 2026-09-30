"""Quick-look replay probe: greedy actor vs JAX flee bot (all device-side), N games.
Prints per-game event timeline (bombs, threat bombs, kills, self-kills, death tick)
plus aggregate 'degenerate behaviour' flags (idle ratio, bomb spam without threat,
oscillation). Used only as a stop/continue sanity check for long runs."""
import os, sys, json, pickle
os.environ.setdefault("JAXBOMB_RULE", "bun")
sys.path.insert(0, "/mnt/jpfs/afs/wangyaqi/code_room/qqt-reward-v2")
import jax, jax.numpy as jnp, numpy as np
from jax_bomb import bun_env as env, jax_train
from jax_bomb.jax_net import transformer_forward
from jax_bomb.bun_frozen_opponents import clear_destructible_bricks

env.prepare()
env.configure_training("danger_arena=1", 1, reward_profile="danger_arena",
                       tactical_bomb_placement_reward=1.0, tactical_bomb_resolution_reward=1.0,
                       base_bomb_reward=1.0, forced_kill_reward=1.0, enemy_threat_reward=1.0)

def run(path, games=256, seed=20261003, steps=300):
    with open(path, "rb") as f:
        v = pickle.load(f)
    params = jax.tree.map(jnp.asarray, v.get("params", v))

    @jax.jit
    def episode(states, key):
        def body(carry, _):
            states, key, active = carry
            key, kb, ks = jax.random.split(key, 3)
            obs = jax_train.both_perspectives(states)[:games]
            gs = jax_train.both_states(states)[:games]
            mm, bm = jax_train.both_masks(states)
            ml, al, _, _ = transformer_forward(params, obs, gs)
            joint = jax_train.adjusted_joint_logits(ml, al, mm[:games], bm[:games],
                jnp.zeros((games, env.N_MOVES, env.N_BOMB), jnp.bool_), "off", 0.0)
            a = jnp.argmax(joint, -1)
            actor = jnp.stack([a // env.N_BOMB, a % env.N_BOMB], -1)
            bot = jax_train.flee_bot_actions(states.pos[:, 1], states.pos[:, 0],
                                             mm[games:], bm[games:], kb)
            acts = jnp.stack([actor, bot], 1)
            nxt, done, info = jax.vmap(lambda s, x, k: env.step(s, x, k, auto_reset=False,
                return_info=True))(states, acts, jax.random.split(ks, games))
            keep = active & ~done
            merged = jax.tree.map(lambda o, n: jnp.where(
                keep.reshape((-1,) + (1,) * (o.ndim - 1)), n, o), states, nxt)
            rec = dict(active=active, move=actor[:, 0], bomb=info["bomb_placed"][:, 0],
                       threat=info["threat_bomb_placed"][:, 0],
                       forced=info["forced_kill_created"][:, 0],
                       kill=info["surviving_kill"][:, 0], own=info["own_bomb_defeat"][:, 0],
                       died=info["death"][:, 0], mutual=info["mutual_death"])
            return (merged, key, keep), rec
        _, rec = jax.lax.scan(body, (states, key, jnp.ones((games,), bool)), None, length=steps)
        return rec

    key = jax.random.PRNGKey(seed)
    states = clear_destructible_bricks(env.init_batch(key, games))
    rec = jax.tree.map(np.asarray, episode(states, key))
    act = rec["active"]
    m = lambda k: (rec[k] & act).sum(0)
    length = act.sum(0)
    idle = ((rec["move"] == 4) & act).sum(0) / np.maximum(length, 1)
    osc = ((rec["move"][1:] != rec["move"][:-1]) & np.isin(rec["move"][1:] + rec["move"][:-1], [1, 5]) & act[1:]).sum(0) / np.maximum(length, 1)
    bombs, threat = m("bomb"), m("threat")
    out = {
        "games": games, "seed": seed, "checkpoint": path,
        "surviving_kill_rate": float((m("kill") > 0).mean()),
        "own_bomb_rate": float((m("own") > 0).mean()),
        "mutual_rate": float((m("mutual") > 0).mean()),
        "avg_bombs": float(bombs.mean()), "avg_threat_bombs": float(threat.mean()),
        "threat_share_of_bombs": float(threat.sum() / max(bombs.sum(), 1)),
        "forced_kill_game_rate": float((m("forced") > 0).mean()),
        "avg_len": float(length.mean()), "idle_ratio": float(idle.mean()),
        "oscillation_ratio": float(osc.mean()),
        "spam_games(>=40 bombs,0 kill)": int(((bombs >= 40) & (m("kill") == 0)).sum()),
        "timeline_first3": [],
    }
    for g in range(3):
        ev = []
        for t in range(int(length[g])):
            for k in ("threat", "forced", "kill", "own", "mutual"):
                if rec[k][t, g]:
                    ev.append(f"t{t}:{k}")
        out["timeline_first3"].append({"len": int(length[g]), "bombs": int(bombs[g]), "events": ev[:25]})
    return out

if __name__ == "__main__":
    res = [run(p) for p in sys.argv[1:]]
    print(json.dumps(res, indent=1))
