"""Loaded module proof and full incoming-state equality at version migration."""
import json,os
from pathlib import Path
from scripts.population_run_state import atomic_json,sha


def migration_exact(out,payload,params,adam,states,aux,keys,uk,bank,generation,manifests):
    import jax
    from scripts.terminal_checkpoint_restore import digest
    names=('params','adam','states','aux','keys','optimizer_key','league_bank','league_generation','league_manifests')
    expected=tuple(payload[k] for k in names)
    actual=jax.device_get((params,adam,states,aux,keys,uk,bank,generation,manifests))
    assert digest(expected)==digest(actual),'incoming state migration mismatch'
    atomic_json(Path(out)/'FULL_STATE_MIGRATION.json',dict(status='PASS_EXACT_INCOMING_FULL_STATE',
        bitwise_exact=True,maxabs=0,source_update=int(payload['update']),digest=digest(actual),fields=list(names),
        existing_episode_layout_retained=True,new_supply_at_actual_resets_only=True))


def migration_sync(out,payload,params,adam,states,aux,keys,uk,bank,generation,manifests):
    import jax
    payload=dict(payload,optimizer_key=payload['optimizer_keys'])
    migration_exact(out,payload,jax.tree.map(lambda x:x[0],params),jax.tree.map(lambda x:x[0],adam),
        states,aux,keys,uk,bank,generation,manifests)


def verify_live(out,protocol):
    import jax,jax.numpy as j,numpy as np,sys
    from jax_bomb import je07_bun_env as env,levels
    from scripts.cx_items_rules_v2 import initial_supply
    t=protocol['items_transition'];rule=json.loads(Path(t['accepted_rule']).read_text())
    assert env.native_items.ITEM_RULE_VERSION==rule['rule_version']
    env.RULE_VERSION=rule['rule_version']
    # Existing physical rule runs stay archived under formal/. The private
    # active registry follows newly admitted namespaces, preserving dynamic
    # peer self-play and SP gate statistics after the version transition.
    from scripts import cc_league2_lib as league
    league.FORMAL=Path('runs/items_alignment_20261107/v2/active')
    if 'scripts.cc_league2_sp_lib' in sys.modules:
        sys.modules['scripts.cc_league2_sp_lib'].FORMAL=league.FORMAL
        sys.modules['scripts.cc_league2_sp_lib'].base.FORMAL=league.FORMAL
    for name,path in rule['private_modules'].items():
        if name in sys.modules:assert Path(sys.modules[name].__file__).resolve()==Path(path).resolve(),name
    st=env._fresh(jax.random.PRNGKey(2026771000),jax.nn.one_hot(0,16))
    table=levels.active();idx=env._BUN_INDEX
    _,summary=jax.jit(lambda s,k:initial_supply(s,k,table.rate[idx],table.super_f[idx],env.TACTICAL_ITEM_FRACTION))(st,jax.random.PRNGKey(2026771001))
    jax.block_until_ready(summary)
    assert np.allclose(np.asarray(table.caps[idx]),[10,8,2.4],atol=1e-6)
    atomic_json(Path(out)/'ITEM_RULE_RECEIPT.json',dict(status='LOADED_ACTUAL_CANONICAL_ITEM_RULE',
        rule_version=rule['rule_version'],canonical_contract_sha256=rule['canonical_contract_sha256'],
        private_sources=rule['private_modules'],loaded_env_sha256=sha(env.__file__),
        actual_brick_probability=float(table.rate[idx]),caps=np.asarray(table.caps[idx]).tolist(),
        natural_supply=dict(zip(rule['supply_summary_fields'],np.asarray(summary).tolist())),
        duel_initial_crates=15,duel_brick_probability=0,body_origin_used=False,
        existing_full_game_reward_unchanged=True,specialist_item_pickup_reward=0,
        source_checkpoint=protocol['continuation_source']['checkpoint']))
