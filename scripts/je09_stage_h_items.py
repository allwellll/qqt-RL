"""Native carry-item events, episode deduplication and bounded settlement."""
import jax
import jax.numpy as jnp
from jax_bomb import je07_bun_env as env
from scripts.je08_bun_potential import shortest, INF

METRICS = ('glue_placed', 'glue_qualified_placement', 'glue_attributed_hit', 'banana_start',
           'banana_end', 'banana_before', 'banana_after', 'banana_gain', 'item_positive',
           'item_clawback', 'item_capture', 'item_alive_carry', 'item_net')
CAP = 2.


def ledger_state(n):
    z = jnp.zeros((n, 2))
    return dict(custody=z.astype(bool), gross=z, balance=z, slide=z.astype(bool),
                slide_distance=z, slide_valid=z.astype(bool))


def attributable_glue(recorded, same_owner, native_hit, slow_edge, free_enemy, safe_carry):
    return recorded & same_owner & native_hit & slow_edge & free_enemy & safe_carry


def settle(e, s):
    eligible = s['custody'] & e['carry'] & ~e['bad'] & e['safe']
    start = e['banana_start'] & eligible & ~s['slide']
    slide = s['slide'] | start
    valid = jnp.where(start, True, s['slide_valid']) & eligible & ~e['topology_changed']
    initial = jnp.where(start, e['distance'], s['slide_distance'])
    finish = slide & ~e['sliding']
    gain = jnp.maximum(initial-e['distance'], 0)
    banana = finish & valid & (initial < INF) & (e['distance'] < INF) & (gain > 0)
    glue = e['glue_hit'] & eligible
    positive = jnp.minimum(.5*glue + jnp.where(banana, jnp.minimum(.5, .1*gain), 0),
                           jnp.maximum(0, CAP-s['gross']))
    balance = s['balance'] + positive
    lost = e['bad'] | (~e['carry'] & ~e['capture'])
    clawback = jnp.where(lost, -balance, 0)
    new = dict(custody=(s['custody'] | e['acquired']) & e['carry'] & ~e['bad'],
               gross=s['gross']+positive, balance=jnp.where(lost | e['capture'], 0, balance),
               slide=slide & e['sliding'] & ~lost, slide_distance=initial,
               slide_valid=valid & ~finish & ~lost)
    metrics = dict(glue_attributed_hit=glue.astype(jnp.float32), banana_start=start.astype(jnp.float32),
                   banana_end=finish.astype(jnp.float32), banana_before=jnp.where(finish, initial, 0),
                   banana_after=jnp.where(finish, e['distance'], 0),
                   banana_gain=jnp.where(banana, gain, 0), item_positive=positive,
                   item_clawback=clawback, item_capture=e['capture'].astype(jnp.float32),
                   item_alive_carry=eligible.astype(jnp.float32), item_net=positive+clawback)
    return positive+clawback, new, metrics


def state(n):
    return dict(ledger=ledger_state(n), glue_token=jnp.zeros((n, 2, env.H, env.W), jnp.int32),
                nonce=jnp.zeros((n, 2), jnp.int32),
                banana_cell=jnp.full((n, 2), -1, jnp.int32),
                banana_owner=jnp.full((n, 2), -1, jnp.int32),
                banana_tick=jnp.full((n, 2), -1, jnp.int32),
                ready=jnp.zeros(n, bool), wall=jnp.zeros((n, env.H, env.W), bool),
                brick=jnp.zeros((n, env.H, env.W), bool),
                distance=jnp.full((n, 2, env.H, env.W), INF, jnp.int32))


def transition(before, after, info, actions, s, goals, terminal):
    n = before.pos.shape[0]; rows = jnp.arange(n)[:, None]; teams = jnp.arange(2)[None]
    enemy = 1-teams
    oldcell = jnp.clip(before.pos.astype(jnp.int32), jnp.array([0, 0]), jnp.array([env.H-1, env.W-1]))
    cell = jnp.clip(after.pos.astype(jnp.int32), jnp.array([0, 0]), jnp.array([env.H-1, env.W-1]))
    changed = jnp.any((before.wall != after.wall) | (before.brick != after.brick), (-2, -1))
    dirty = ~s['ready'] | changed
    blocked = after.wall | after.brick
    allgoals = jnp.broadcast_to(goals, (n, 2, env.H, env.W))
    distance = jax.lax.cond(jnp.any(dirty),
        lambda _: jnp.where(dirty[:, None, None, None],
                            shortest(blocked[:, None], jnp.zeros_like(blocked[:, None]), allgoals), s['distance']),
        lambda _: s['distance'], None)
    d = distance[rows, teams, cell[..., 0], cell[..., 1]]
    # Banana event begins before its first forced displacement.
    oldd = distance[rows, teams, oldcell[..., 0], oldcell[..., 1]]
    carried = (before.bun_carried == enemy) & (after.bun_carried == enemy)
    safe_carry = carried & before.alive & after.alive & (before.trapped == 0) & (after.trapped == 0)
    bad = info['death'] | info['drop'] | info['carrier_death'] | info['mutual_death'][:, None] | (after.trapped > 0)
    owned = s['ledger']['custody'] & safe_carry & ~bad & ~terminal[:, None]
    slot = before.item_slots[:, :, 0]
    prior = before.field_item[rows, oldcell[..., 0], oldcell[..., 1]]
    placed = (actions[:, :, 1] == 2) & (slot == env.ITEM_SLOW_GLUE) & (prior == 0)
    oldcount = jnp.sum(jnp.where(before.item_slots == env.ITEM_SLOW_GLUE, before.item_counts, 0), -1)
    newcount = jnp.sum(jnp.where(after.item_slots == env.ITEM_SLOW_GLUE, after.item_counts, 0), -1)
    placed &= newcount < oldcount
    placed &= (after.field_item[rows, oldcell[..., 0], oldcell[..., 1]] == env.ITEM_SLOW_GLUE)
    placed &= (after.field_owner[rows, oldcell[..., 0], oldcell[..., 1]] == teams)
    # Treat glue as blocked for its owner: no reward for trapping the carrier
    # or placing a field on the only legal path into our cage.
    safety_blocked = blocked | (after.fuse > 0) | (after.core.blast_linger > 0) | (after.field_item == env.ITEM_SLOW_GLUE)
    need_safety = jnp.any(placed) | jnp.any(s['ledger']['slide']) | jnp.any(info['trap_hit'])
    safety_distance = jax.lax.cond(need_safety,
        lambda _: shortest(safety_blocked[:, None], jnp.zeros_like(safety_blocked[:, None]), allgoals),
        lambda _: distance, None)
    safe = (safety_distance[rows, teams, cell[..., 0], cell[..., 1]] < INF)
    neighbors = jnp.clip(cell[:, :, None]+jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]]),
                         jnp.array([0, 0]), jnp.array([env.H-1, env.W-1]))
    exits = safety_distance[rows[:, :, None], teams[:, :, None], neighbors[..., 0], neighbors[..., 1]]
    on_unarmed_glue = (after.field_item[rows, cell[..., 0], cell[..., 1]] == env.ITEM_SLOW_GLUE)
    on_unarmed_glue &= ~after.field_armed[rows, cell[..., 0], cell[..., 1]]
    safe |= on_unarmed_glue & jnp.any(exits < INF, -1)
    safe &= ~terminal[:, None] & ~changed[:, None]
    qualified = placed & owned & safe & ~allgoals[rows, teams, oldcell[..., 0], oldcell[..., 1]]
    nonce = s['nonce'] + qualified.astype(jnp.int32)
    # A map token exists only for an actual eligible placement in this episode.
    tokens = s['glue_token']
    hitcell = cell[:, ::-1]
    token = tokens[rows, teams, hitcell[..., 0], hitcell[..., 1]]
    hititem = before.field_item[rows, hitcell[..., 0], hitcell[..., 1]]
    hitowner = before.field_owner[rows, hitcell[..., 0], hitcell[..., 1]]
    free = before.alive[:, ::-1] & after.alive[:, ::-1] & (before.trapped[:, ::-1] == 0) & (after.trapped[:, ::-1] == 0)
    slowedge = (after.move_status[:, ::-1] == env.STATUS_SLOW) & (before.move_status[:, ::-1] != env.STATUS_SLOW)
    glue = attributable_glue(token > 0, (hitowner == teams) & (hititem == env.ITEM_SLOW_GLUE),
                            info['trap_hit'][:, ::-1] & (info['trap_owner'][:, ::-1] == teams),
                            slowedge, free, owned & safe)
    still = (after.field_item[:, None] == env.ITEM_SLOW_GLUE) & (after.field_owner[:, None] == teams[:, :, None, None])
    tokens = jnp.where(still, tokens, 0)
    previous = tokens[rows, teams, oldcell[..., 0], oldcell[..., 1]]
    tokens = tokens.at[rows, teams, oldcell[..., 0], oldcell[..., 1]].set(jnp.where(qualified, nonce, previous))
    source = before.field_item[rows, cell[..., 0], cell[..., 1]]
    source_owner = before.field_owner[rows, cell[..., 0], cell[..., 1]]
    banana = info['trap_hit'] & (source == env.ITEM_BANANA) & (after.move_status == env.STATUS_SLIDE)
    banana &= before.field_armed[rows, cell[..., 0], cell[..., 1]] & (source_owner == info['trap_owner'])
    acquired = ((info['pickup_loose'] & (info['pickup_origin'] == enemy)) | info['steal'] | info['native_transfer'])
    acquired &= (before.bun_carried < 0) & (after.bun_carried == enemy) & after.alive
    e = dict(carry=after.alive & (after.bun_carried == enemy), acquired=acquired,
             bad=bad, capture=info['capture'], glue_hit=glue, banana_start=banana,
             sliding=after.move_status == env.STATUS_SLIDE,
             distance=jnp.where(banana, oldd, d), safe=safe, topology_changed=changed[:, None])
    reward, ledger, metrics = settle(e, s['ledger'])
    metrics.update(glue_placed=placed.astype(jnp.float32), glue_qualified_placement=qualified.astype(jnp.float32))
    new = dict(ledger=ledger, glue_token=tokens, nonce=nonce,
               banana_cell=jnp.where(banana, cell[..., 0]*env.W+cell[..., 1], s['banana_cell']),
               banana_owner=jnp.where(banana, source_owner, s['banana_owner']),
               banana_tick=jnp.where(banana, before.t[:, None], s['banana_tick']),
               ready=jnp.ones(n, bool), wall=after.wall, brick=after.brick, distance=distance)
    return reward, new, metrics
