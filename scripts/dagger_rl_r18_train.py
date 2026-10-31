"""R18 three-arm DAgger/RL schedule with certified item-chain route labels.
The only treatment is removal of ability=2 from route supervision when no
complete item plan is certified; rewards, masks, optimizer and exposure stay frozen."""
import sys
import pickle
import subprocess
import numpy as np
import jax
import optax
from scripts import dagger_rl_r12_train as L
from scripts.dagger_rl_r18_run import B, R, read, authorize, notify, UserStopped
from scripts.dagger_rl_item_chain_teacher import compose_teacher
from scripts import dagger_rl_release_qualification as _qualification
from scripts.dagger_rl_retention_schedule import make_update
from scripts.dagger_rl_retention_loop import formal_function
from scripts.carrier_mechanism_train import preserve_critic
from scripts.terminal_checkpoint_restore import digest
from scripts.population_run_state import atomic_json, sha

BASE_BIND = L.bind
ITEM_CHAIN_INSTALLED = False


def install_item_chain_teacher():
    global ITEM_CHAIN_INSTALLED
    if ITEM_CHAIN_INSTALLED:
        return
    # r10's frozen collector calls this module-level alias; patch the same
    # callable used by the direct collector and auxiliary student stream.
    _qualification.combine = compose_teacher
    L.C.O.combine = compose_teacher
    ITEM_CHAIN_INSTALLED = True


def bind():
    L.B, L.R, L.read, L.authorize, L.notify = B, R, read, authorize, notify
    BASE_BIND()
    install_item_chain_teacher()


def updates(optimizer):
    return dict(joint_anchor=L.C.updater('anchor', optimizer),
                dagger_anchor=make_update(L.C.objective, optimizer, 'dagger_anchor', preserve_critic),
                ppo_anchor=make_update(L.C.objective, optimizer, 'ppo_anchor', preserve_critic))


def probe(x, bs, ba):
    p, a = L.rep(x['params']), L.rep(x['adam'])
    aux = x['auxiliary']
    pend, batch, diag = L.C.collector(8)(p, x['states'], x['aux'], x['keys'], bs, ba)
    aend, abatch, adiag = L.auxiliary_collector(8, greedy=False)(
        p, aux['states'], aux['aux'], aux['keys'], bs, ba)
    h, rb, stats = L.T.aggregate(abatch, x['history'], x['seed'], x['update']+1)
    rb = tuple(v[:, :64] for v in rb)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5, eps=1e-8))
    out = {}
    for branch, update in updates(opt).items():
        (pn, an), loss = update(p, a, batch, rb, np.asarray(jax.random.split(jax.random.PRNGKey(81), 8)))
        jax.block_until_ready(pn)
        out[branch] = digest((L.one(pn), L.one(an), pend, aend, batch, abatch, h, rb, stats, np.asarray(loss)))
    return out


def smoke():
    bind()
    items, source = L.T.setup()
    seed = 2026102691
    st, ax, key, bs, ba = L.T.initial(seed, items)
    ast, aax, akey, _, _ = L.T.initial(seed ^ 0xAA12, items)
    p, a = L.rep(source['params']), L.rep(source['adam'])
    pend, batch, diag = L.C.collector(8)(p, st, ax, key, bs, ba)
    aend, abatch, adiag = L.auxiliary_collector(8, greedy=False)(p, ast, aax, akey, bs, ba)
    h, rb, stats = L.T.aggregate(abatch, None, seed, 1)
    rb = tuple(v[:, :64] for v in rb)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(7.5e-5, eps=1e-8))
    rows, outputs = {}, {}
    for branch, update in updates(opt).items():
        (pn, an), loss = update(p, a, batch, rb, np.asarray(jax.random.split(jax.random.PRNGKey(8), 8)))
        jax.block_until_ready(pn)
        now, adam = L.one(pn), L.one(an)
        delta = L.norms(source['params'], now)
        ability = float(optax.global_norm(jax.tree.map(lambda x, y: x-y,
            now['actor']['heads']['wb'], source['params']['actor']['heads']['wb'])))
        assert delta['actor'] > 0 and ability > 0 and float(np.asarray(loss)[..., -1].mean()) > 0
        assert (delta['critic'] == 0) == (branch == 'dagger_anchor')
        assert np.isfinite(np.asarray(loss)).all() and all(np.isfinite(v).all() for v in jax.tree.leaves((now, adam)))
        if branch == 'dagger_anchor':
            for name in ['mu', 'nu']:
                assert digest(getattr(adam[1][0], name)['critic']) == digest(getattr(source['adam'][1][0], name)['critic'])
        rows[branch] = dict(finite=True, nonzero=delta, ability_update_L2=ability,
                            ability_raw_gradient_norm=float(np.asarray(loss)[..., -1].mean()),
                            critic_intentionally_frozen=branch == 'dagger_anchor')
        outputs[branch] = pn, an
    pn, an = outputs['joint_anchor']
    cp = L.commit(B/'smoke', 1, pn, an, *pend, seed, 'technical', h, formal=False, auxiliary=aend)
    x = pickle.load((cp/'state.pkl').open('rb'))
    expected = probe(x, bs, ba)
    atomic_json(B/'smoke/EXPECTED.json', dict(next_branches=expected, state_sha256=sha(cp/'state.pkl')))
    with (B/'smoke/restore.log').open('a') as f:
        code = subprocess.run([sys.executable, '-m', 'scripts.dagger_rl_r18_train', 'restore'],
                              stdout=f, stderr=subprocess.STDOUT).returncode
    assert code == 0 and read(B/'smoke/RESTORE.json')['passed']
    atomic_json(B/'TECHNICAL_SMOKE.json', dict(passed=True, branches=rows,
        fresh_full_both_stream_restore=True, formal_inherits_smoke=False,
        total_executed_environment_steps=3072, total_executed_optimizer_steps=18,
        actual_branch_optimizer_steps=6, reference_optimizer_steps=6, fresh_restore_optimizer_steps=6,
        adopted_formal_steps=0, capability_gate=False))
    notify('R18技术smoke通过：三损失分支真实Actor/ability非零、PPO Critic非零、DAgger Critic及moments冻结；双流完整fresh恢复精确，立即正式固定队列。')


def restore():
    bind()
    items, _ = L.T.setup()
    _, _, _, bs, ba = L.T.initial(2026102691, items)
    expected = read(B/'smoke/EXPECTED.json')
    cp = B/'smoke/u001'
    assert sha(cp/'state.pkl') == expected['state_sha256']
    actual = probe(pickle.load((cp/'state.pkl').open('rb')), bs, ba)
    if actual != expected['next_branches']:
        atomic_json(B/'smoke/RESTORE_MISMATCH.json', dict(passed=False,
            expected=expected['next_branches'], actual=actual,
            state_sha256=sha(cp/'state.pkl'),
            note='diagnostic only; formal launch remains blocked'))
        raise AssertionError('fresh restore branch hashes differ; see RESTORE_MISMATCH.json')
    atomic_json(B/'smoke/RESTORE.json', dict(passed=True, full_next_branch_hashes=actual))


def formal():
    arm, seed = sys.argv[2], int(sys.argv[3])
    p = read(B/'PROTOCOL.json')
    assert arm in p['arms'] and seed in p['paired_seeds']
    bind()
    formal_function(L, bind=bind, base=B, root=R, read=read, authorize=authorize,
                    notify=notify, arm=arm, seed=seed)()


if __name__ == '__main__':
    try:
        authorize()
        globals()[sys.argv[1]]()
    except UserStopped:
        atomic_json(B/'PAUSED_WORKER.json', dict(safe_atomic=True, old_continuous_resume=False))
