import numpy as np
import jax
import jax.numpy as jnp
from scripts.return_query_diversity import first_deviations
from scripts.prestock_fragment_queries import first_deviations as baseline


def run(fragment,cells,*,mismatch=None,terminal=None,preparing=None,revision=None,opening=None):
    n=len(fragment)
    expand=lambda x,dtype:jnp.asarray(x,dtype)[:,None]
    values=[expand(mismatch if mismatch is not None else [True]*n,bool),
            expand(fragment,jnp.int32),expand(terminal if terminal is not None else [False]*n,bool),
            expand(preparing if preparing is not None else [False]*n,bool),
            expand(revision if revision is not None else [0]*n,jnp.int32),
            expand(opening if opening is not None else [False]*n,bool),expand(cells,jnp.int32)]
    return np.asarray(jax.jit(first_deviations)(*values))[:,0]


def test_second_verified_return_deviation_can_cover_a_later_distinct_cell():
    mask=run([3]*24,[80]*8+[81]*8+[82]*8)
    np.testing.assert_array_equal(np.flatnonzero(mask),[0,8])


def test_same_cell_and_short_gap_cannot_mint_return_queries():
    mask=run([3]*24,[80]*12+[81]*12)
    np.testing.assert_array_equal(np.flatnonzero(mask),[0,12])
    mask=run([3]*24,[80,81]+[80]*22)
    np.testing.assert_array_equal(np.flatnonzero(mask),[0])


def test_phase_switch_and_revision_do_not_reset_return_quota():
    fragment=[3]+[2]*7+[3]+[1]*7+[3]*8
    mask=run(fragment,list(range(24)),revision=list(range(24)))
    np.testing.assert_array_equal(np.flatnonzero(mask),[0,1,8,9])


def test_real_terminal_resets_return_quota_after_its_transition():
    end=[False]*24;end[10]=True
    mask=run([3]*24,[80]*8+[81]*3+[80]*8+[81]*5,terminal=end)
    np.testing.assert_array_equal(np.flatnonzero(mask),[0,8,11,19])


def test_uncertified_or_non_deviating_rows_never_become_queries():
    mismatch=[False]*16+[True]*8
    mask=run([3]*24,list(range(24)),mismatch=mismatch)
    np.testing.assert_array_equal(np.flatnonzero(mask),[16])
    assert not run([3]*24,list(range(24)),mismatch=[False]*24).any()


def test_other_fragments_keep_one_query_and_preparation_keeps_revision_quotas():
    for fragment in [1,2,4]:
        np.testing.assert_array_equal(np.flatnonzero(run([fragment]*24,list(range(24)))),[0])
    prep=[True]*16;revision=[0]*8+[1]*8;opening=[False]*4+[True]*4+[False]*4+[True]*4
    np.testing.assert_array_equal(np.flatnonzero(run([2]*16,list(range(16)),preparing=prep,
        revision=revision,opening=opening)),[0,4,8,12])


def test_batched_fixed_states_preserve_every_baseline_first_query():
    rng=np.random.default_rng(20261018);shape=(64,32)
    fragment=rng.integers(0,5,shape,dtype=np.int32)
    mismatch=(rng.random(shape)<.7)&(fragment>0)
    terminal=rng.random(shape)<.03
    preparing=(fragment==2)&(rng.random(shape)<.5)
    revision=np.cumsum(rng.random(shape)<.1,axis=0).astype(np.int32)
    opening=rng.random(shape)<.5;cells=rng.integers(0,195,shape,dtype=np.int32)
    args=list(map(jnp.asarray,[mismatch,fragment,terminal,preparing,revision,opening]))
    old=np.asarray(jax.jit(baseline)(*args));new=np.asarray(jax.jit(first_deviations)(*args,jnp.asarray(cells)))
    assert np.all(~old|new)
    np.testing.assert_array_equal(new[fragment!=3],old[fragment!=3])
    for env in range(shape[1]):
        last=None;count=0
        for tick in range(shape[0]):
            if new[tick,env] and fragment[tick,env]==3:
                assert count<2
                if last is not None:
                    assert tick-last>=8 and cells[tick,env]!=cells[last,env]
                last=tick;count+=1
            if terminal[tick,env]:last=None;count=0
