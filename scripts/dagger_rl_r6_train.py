"""Isolated single-variable opponent exposure; frozen R4 optimization math."""
import json
import os
import pickle
import subprocess
import sys
import time
import numpy as np
import jax
import jax.numpy as j
import optax
from scripts import dagger_rl_r4_train as C
from scripts.dagger_rl_r6_run import B, R, read, authorize, notify
from scripts.dagger_rl_source_replay import aggregate_with_sources
from scripts.dagger_rl_opponent_mix import assign, enemy_inputs, sample_parent, replace_enemy
from scripts.population_run_state import atomic_json, sha
from scripts.terminal_checkpoint_restore import digest

BASE_SETUP = C.setup
BASE_INITIAL = C.initial
BASE_CHOOSE = C.E.choose
ACTIVE_MIX = True
INSTALLED = False


def choose(params, states, aux, keys):
    nk, ke, actions, data = BASE_CHOOSE(params, states, aux, keys)
    o, g, mm, bm = enemy_inputs(C.E.train, states)
    mv, ab, _, _ = C.E.train.net_forward(C.REFERENCE_PARAMS, 'transformer', o, g)
    ko = jax.vmap(lambda key: jax.random.split(key, 5))(keys)[:, 3]
    parent = sample_parent(mv, ab, mm, bm, ko, aux['terminal'])
    selected = aux['opponent_enabled'] & aux['opponent_candidate']
    return nk, ke, replace_enemy(actions, parent, selected), data


def setup():
    global INSTALLED
    items, source = BASE_SETUP()
    if not INSTALLED:
        fresh, transition = C.T.fresh, C.T.transition

        def with_assignment(keys, bs, ba):
            states, aux = fresh(keys, bs, ba)
            return states, dict(aux, opponent_candidate=assign(keys),
                                opponent_enabled=j.zeros(keys.shape[0], bool))

        def with_persistent_assignment(states, aux, actions, keys, arm='control'):
            extra = {k: aux[k] for k in ['opponent_candidate', 'opponent_enabled']}
            old = {k: v for k, v in aux.items() if k not in extra}
            after, na, reward, done, info, parts = transition(states, old, actions, keys, arm)
            return after, dict(na, **extra), reward, done, info, parts

        C.T.fresh = with_assignment
        C.T.transition = with_persistent_assignment
        C.E.choose = choose
        INSTALLED = True
    return items, source


def initial(seed, items):
    st, ax, key, bs, ba = BASE_INITIAL(seed, items)
    ax = dict(ax, opponent_enabled=j.full_like(ax['opponent_candidate'], ACTIVE_MIX))
    return st, ax, key, bs, ba


def collector(steps):
    def collect(p, st, ax, keys, bs, ba):
        def body(c, _):
            st, ax, key = c
            nk, ke, act, data = C.E.choose(p, st, ax, key)
            o, g, mm, bm, own, lp, v = data
            route = C.T.route_teacher(st, ax)
            item, od = C.O.teacher(st, ax)
            teacher = C.O.combine(route, item)
            after, na, reward, done, info, parts = C.T.transition(st, ax, act, ke, 'control')
            nv = C.E.prediction(p, after)[2]
            fresh, fa = C.T.fresh(jax.vmap(lambda k: jax.random.fold_in(k, 0xF3))(nk), bs, ba)
            # Assignment changes at episode reset; the treatment arm never does.
            fa = dict(fa, opponent_enabled=ax['opponent_enabled'])
            after = C.E.select_tree(done, fresh, after)
            na = C.E.select_tree(done, fa, na)
            carry = st.bun_carried[:, 0] == 1
            window = ~st.alive[:, 1] | (st.trapped[:, 1] > 0)
            _, ab, _, _ = C.E.train.net_forward(p, 'transformer', o, g)
            probability = jax.nn.softmax(j.where(bm, ab, -1e9))[:, 2]
            flags = j.stack([carry, st.held_item[:, 0] == 1, bm[:, 2], own[:, 1] == 2,
                j.argmax(j.where(bm, ab, -1e9), -1) == 2, probability,
                info['trap_hit'][:, 0], info['capture'][:, 0], info['death'][:, 0],
                info['steal'][:, 0] & window, info['steal'][:, 0] & ~window,
                info['pickup_loose'][:, 0], info['native_transfer'][:, 0],
                info['own_bomb_defeat'][:, 0], ~carry & ~window, ~carry & window], -1)
            live = ~ax['terminal']
            opponent = ax['opponent_enabled'] & ax['opponent_candidate']
            combat = C.combat_rows(o, g) & live
            distance = j.abs(st.pos[:, 0] - st.pos[:, 1]).sum(-1)
            exposure = j.stack([opponent, live, combat, combat & (distance <= 4),
                                 combat & (distance > 4) & (distance <= 8),
                                 combat & (distance > 8)], -1)
            return (after, na, nk), (o, g, own, lp, v, nv, reward[:, 0], done,
                mm, bm, teacher, item.any((1, 2)), live, parts[:, 0], flags, od,
                ax['provenance_source'], exposure)
        end, d = jax.lax.scan(body, (st, ax, keys), None, length=steps)
        o,g,act,lp,v,nv,reward,done,mm,bm,teacher,isitem,live,parts,flags,od,origins,exposure = d
        advantage = C.E.train.compute_gae(reward, v, nv, done, .995, .95)
        return end, (o,g,act,lp,advantage,advantage+v,mm,bm,teacher,isitem,live,origins), \
            (parts,flags,od,reward,v,advantage,origins,exposure)
    return jax.pmap(collect, in_axes=(0,0,0,0,None,None))


def aggregate(batch, history, seed, update):
    b = tuple(np.asarray(x).reshape((-1,) + x.shape[3:]) for x in jax.device_get(batch))
    new = (b[0],b[1],b[6],b[7],b[8],b[9])
    eligible = b[8].any((1,2))
    accepted = tuple(x[eligible] for x in new)
    h, replay, stats = aggregate_with_sources(accepted, b[11][eligible], history,
        seed=seed, update=update, strategy='reservoir', balance_items=False,
        fallback=new, fallback_sources=b[11])
    return h, tuple(x.reshape((8,1024)+x.shape[1:]) for x in replay), stats


def bind():
    C.B, C.R, C.read = B, R, read
    C.authorize, C.notify = authorize, notify
    C.setup, C.initial, C.collector, C.aggregate = setup, initial, collector, aggregate


def smoke():
    bind()
    items, source = setup()
    st, ax, key, bs, ba = initial(2026102498, items)
    # Actual native states, opponent's own view, no change to learner outputs/RNG.
    local = lambda tree: jax.tree.map(lambda v: v[0], tree)
    args = (source['params'], local(st), dict(local(ax), opponent_enabled=j.zeros(8,bool)), key[0])
    expected = jax.jit(BASE_CHOOSE)(*args)
    actual = jax.jit(choose)(*args)
    assert digest(expected) == digest(actual), 'rule3 control changed'
    treatment = jax.jit(choose)(source['params'], local(st), local(ax), key[0])
    assert digest(expected[:2]) == digest(treatment[:2])
    assert digest(expected[3]) == digest(treatment[3])
    np.testing.assert_array_equal(expected[2][:,0], treatment[2][:,0])
    selected = np.asarray(local(ax)['opponent_candidate'])
    changed = np.any(np.asarray(expected[2][:,1]) != np.asarray(treatment[2][:,1]), -1)
    assert selected.any() and (selected & changed).any(), 'opponent treatment absent in technical batch'
    assert not changed[~selected].any()
    atomic_json(B/'OPPONENT_ACTION_PARITY.json', dict(passed=True,
        control_full_output_exact=True,learner_action_logp_value_RNG_exact=True,
        opponent_selected=int(selected.sum()),opponent_actions_differ=int(changed.sum()),
        frozen_policy_sha256=read(B/'PROTOCOL.json')['parent']['policy_sha256'],
        scope='same real native observations; no environment or adopted update'))
    opt = optax.chain(optax.clip_by_global_norm(.5),optax.adam(7.5e-5,eps=1e-8))
    p,a = C.rep(source['params']),C.rep(source['adam'])
    (st,ax,key),batch,diag = collector(8)(p,st,ax,key,bs,ba)
    history,rb,stats = aggregate(batch,None,2026102498,1)
    rb = tuple(x[:,:64] for x in rb)
    (p,a),loss = C.updater('anchor',opt)(p,a,batch,rb,np.asarray(jax.random.split(jax.random.PRNGKey(8),8)))
    jax.block_until_ready(p)
    changes = C.norms(source['params'],C.one(p))
    ability = float(optax.global_norm(jax.tree.map(lambda x,y:x-y,
        C.one(p)['actor']['heads']['wb'],source['params']['actor']['heads']['wb'])))
    assert all(v>0 for v in changes.values()) and ability>0
    assert np.isfinite(np.asarray(loss)).all()
    assert all(np.isfinite(v).all() for v in jax.tree.leaves((C.one(p),C.one(a))))
    cp = C.commit(B/'smoke',1,p,a,st,ax,key,2026102498,'technical',history,formal=False)
    x = pickle.load((cp/'state.pkl').open('rb'))
    continuation = C.next_update_probe(x,bs,ba)
    atomic_json(B/'smoke/EXPECTED.json',dict(next_update=continuation,
        adam=digest(x['adam']),history=digest(history),nonzero=changes))
    code = subprocess.run([sys.executable,'-m','scripts.dagger_rl_r6_train','restore'],
        stdout=(B/'smoke/restore.log').open('a'),stderr=subprocess.STDOUT).returncode
    assert code==0 and read(B/'smoke/RESTORE.json')['passed']
    atomic_json(B/'TECHNICAL_SMOKE.json',dict(passed=True,finite=True,nonzero=changes,
        ability_head_update_l2=ability,ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),
        fresh_process_restore=True,parity_environments=64,opponent_metadata_exact=True,
        environment_steps=512,optimizer_steps=2,total_executed_environment_steps=1536,
        total_executed_optimizer_steps=6,formal_inherits_smoke=False,capability_gate=False,
        state_sha256=sha(cp/'state.pkl'),aggregation=stats))
    notify('R6技术smoke通过：同状态control及learner RNG/logp精确、冻结父对手实际动作不同；finite/非零及fresh64下一完整更新精确。立即正式队列。')


def restore():
    bind()
    C.restore()


rep, one, norms = C.rep, C.one, C.norms
save, commit, updater = C.save, C.commit, C.updater


def formal():
 global ACTIVE_MIX
 bind()
 items,source=setup();pr=read(B/'PROTOCOL.json');opt=optax.chain(optax.clip_by_global_norm(.5),optax.adam(pr['lr'],eps=1e-8));updates={m:updater(m,opt) for m in ['joint','anchor']};collect=collector(128)
 control=B/'formal/no_update/u000';control.mkdir(parents=True,exist_ok=True)
 if not (control/'COMMITTED.json').exists():
  save(control/'state.pkl',source);save(control/'policy.pt',source['params']['actor']);atomic_json(control/'COMMITTED.json',dict(passed=True,update=0,no_update=True,state_sha256=sha(control/'state.pkl'),policy_sha256=sha(control/'policy.pt')))
 records=[]
 for seed in pr['paired_seeds']:
  for arm in pr['arms']:
   ACTIVE_MIX=arm=='parent_mix'
   authorize();base=B/'formal'/f'{arm}_s{seed}';base.mkdir(parents=True,exist_ok=True)
   if (base/'COMPLETE.json').exists():records.append(read(base/'COMPLETE.json'));continue
   p=rep(source['params']);a=rep(source['adam']);st,ax,key,bs,ba=initial(seed,items);start=0;history=None
   last=sorted(base.glob('u*/COMMITTED.json'))
   if last:
    cp=last[-1].parent;assert sha(cp/'state.pkl')==read(cp/'COMMITTED.json')['state_sha256'];x=pickle.load((cp/'state.pkl').open('rb'));start=x['update'];p=rep(x['params']);a=rep(x['adam']);st=x['states'];ax=x['aux'];key=x['keys'];history=x['history']
   else:commit(base,0,p,a,st,ax,key,seed,arm)
   if start==0:
    assert digest(one(p))==digest(source['params']) and digest(one(a))==digest(source['adam'])
    atomic_json(base/'FRESH_PARENT.json',dict(passed=True,parent_input_sha256=sha(B/'inputs/parent.pkl'),params_exact=True,Adam_exact=True,smoke_not_inherited=True))
   began=time.time()
   for u in range(start+1,33):
    authorize();before=one(p);mode='anchor'
    (st,ax,key),batch,diag=collect(p,st,ax,key,bs,ba);history,rb,ag=aggregate(batch,history,seed,u)
    (p,a),loss=updates[mode](p,a,batch,rb,np.asarray(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),u),8)));jax.block_until_ready(p)
    after=one(p);n=norms(before,after);assert n['actor']>0 and (n['critic']==0 if mode=='dagger' else n['critic']>0)
    assert np.isfinite(np.asarray(loss)).all() and all(np.isfinite(v).all() for v in jax.tree.leaves(after))
    fn=['carry','banana_inventory','ability2_legal','sample_release','greedy_release','release_probability_sum','self_contact','delivery','death','window_stock','free_stock','loose','transfer','self_defeat','empty_combat','empty_window']
    flags=np.asarray(diag[1]);od=np.asarray(diag[2]);item=np.asarray(batch[9]);carry=flags[...,0]>0
    row=dict(arm=arm,seed=seed,update=u,mode=mode,finite=True,update_l2=n,environment_steps=u*8192,optimizer_steps=u*32,aggregation=ag,elapsed_seconds=time.time()-began,effective_steps_this_process=(u-start)*8192,steps_per_second=(u-start)*8192/(time.time()-began),loss=np.asarray(loss).mean((0,1)).tolist(),ability_head_gradient_norm=float(np.asarray(loss)[...,-1].mean()),ability_head_update_l2=float(optax.global_norm(jax.tree.map(lambda x,y:x-y,after['actor']['heads']['wb'],before['actor']['heads']['wb']))),events=dict(zip(fn,flags.sum((0,1,2)).tolist())),item_opportunities=dict(release=int(od[...,2].sum()),contact=int(od[...,3].sum()),mean_forecast_net=float(od[...,4][item].mean()) if item.any() else None),carry_inventory_ticks=int((carry&(flags[...,1]>0)).sum()),carry_release=int((carry&(flags[...,3]>0)).sum()),reward_components=np.asarray(diag[0]).sum((0,1,2)).tolist(),opponent_exposure=dict(current=0,ring=0,fixed=8192,kind='JAX native tier3'),GAE_mean=float(np.asarray(diag[-1]).mean()),GAE_variance=float(np.asarray(diag[-1]).var()))
    exposure=np.asarray(diag[7]);selected=exposure[...,0]>0
    row['opponent_exposure']=dict(current=0,ring=0,fixed=8192,rule3=int((~selected).sum()),historical_parent=int(selected.sum()),kind='JAX rule3/frozen common-parent categorical; fixed historical, not current/ring/JS')
    row['combat_support']={name:dict(rule3=int(((exposure[...,i]>0)&~selected).sum()),historical_parent=int(((exposure[...,i]>0)&selected).sum())) for i,name in enumerate(['unused','live','combat','near_le4','mid_4to8','far_gt8']) if i>0}
    row['GAE_mean']=float(np.asarray(diag[5]).mean());row['GAE_variance']=float(np.asarray(diag[5]).var())
    row['anchor_KL']=float(np.asarray(loss)[...,7].mean());row['anchor_eligible_fraction']=float(np.asarray(loss)[...,8].mean())
    origins=np.asarray(diag[6]);row['source_exposure']={}
    for name,valid in [('natural',origins<0),('real_pickup_bank',origins>=0)]:
     row['source_exposure'][name]=dict(ticks=int(valid.sum()),events={k:float(flags[...,i][valid].sum()) for i,k in enumerate(fn)},item_release_labels=int(od[...,2][valid].sum()),item_contact_labels=int(od[...,3][valid].sum()))
    with (base/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    # Every update is atomic for a safe user stop; evaluator consumes declared points only.
    cp=commit(base,u,p,a,st,ax,key,seed,arm,history,row)
    atomic_json(B/'STATUS.json',dict(stage='FORMAL',**row))
    if u==1:notify(f'R6正式首更新 {arm}/s{seed} u1 {mode} finite：Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，8192真实学生步，{ag["new_item_rows"]}道具老师行，原子SHA {sha(cp/"state.pkl")[:16]}。')
    if u==17:notify(f'R6 {arm}/s{seed} u17：{mode}，Actor Δ{n["actor"]:.6g} Critic Δ{n["critic"]:.6g}，中段邻点完整保存（本轮无阶段切换）；未使用历史数据作PPO。')
   final=base/'u032';receipt=dict(arm=arm,seed=seed,update=32,finite=True,state_sha256=sha(final/'state.pkl'),policy_sha256=sha(final/'policy.pt'),environment_steps=262144,optimizer_steps=1024)
   atomic_json(base/'COMPLETE.json',receipt);records.append(receipt);notify(f'DAgger/RL固定run终态 {arm}/s{seed} u32完整原子提交；262144环境步/1024优化步，立即接续冻结队列，CPU不阻塞。')
 atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=1572864,optimizer_steps=6144,old_continuous_resume=False))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
