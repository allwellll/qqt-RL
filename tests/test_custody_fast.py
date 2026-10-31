import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import jax
import jax.numpy as j
import numpy as np
from scripts.bun_native_v2_custody_fast import sparse_fields
from scripts.je08_bun_potential import shortest,INF


def test_sparse_fields_all_dirty_clean_and_padding_exact():
    n=7;blocked=j.zeros((n,13,15),bool).at[1,:,7].set(True)
    goals=j.zeros((n,2,13,15),bool).at[:,0,1,2].set(True).at[:,1,10,12].set(True)
    expected=shortest(blocked[:,None],j.zeros_like(blocked[:,None]),goals)
    old=j.full_like(expected,INF)
    fn=jax.jit(sparse_fields)
    np.testing.assert_array_equal(fn(blocked,goals,old,j.ones(n,bool)),expected)
    np.testing.assert_array_equal(fn(blocked,goals,expected,j.zeros(n,bool)),expected)
    old=old.at[0].set(expected[0]);dirty=j.arange(n)>0
    np.testing.assert_array_equal(fn(blocked,goals,old,dirty),expected)


def test_single_dirty_row_cannot_overwrite_clean_row_zero():
    n=3;blocked=j.zeros((n,13,15),bool);goals=j.zeros((n,2,13,15),bool).at[2,:,5,5].set(True)
    previous=j.ones((n,2,13,15),j.int32)*27
    out=jax.jit(sparse_fields)(blocked,goals,previous,j.array([False,False,True]))
    np.testing.assert_array_equal(out[:2],previous[:2]);assert int(out[2,0,5,5])==0
