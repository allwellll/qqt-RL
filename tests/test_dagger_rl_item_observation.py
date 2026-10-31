from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from scripts.dagger_rl_item_observation import align_item_presence, aligned_encoder


def test_tactical_pickup_presence_matches_native_union_without_field_presence():
    obs=j.zeros((24,3,4)).at[7,0,0].set(1).at[20,2,3].set(1)
    tactical=j.zeros((3,4),j.int8).at[1,1].set(1).at[1,2].set(2)
    got=jax.jit(lambda o,t:align_item_presence(o,t,enabled=True))(obs,tactical)
    np.testing.assert_array_equal(got[7],np.asarray([[1,0,0,0],[0,1,1,0],[0,0,0,0]]))
    np.testing.assert_array_equal(np.delete(got,7,0),np.delete(obs,7,0))
    assert got.dtype==obs.dtype
    np.testing.assert_array_equal(obs[7],np.asarray([[1,0,0,0],[0,0,0,0],[0,0,0,0]]))


def test_disabled_is_exact_legacy_and_alignment_is_idempotent():
    obs=j.arange(24*3*4,dtype=j.float32).reshape(24,3,4)%2
    tactical=j.ones((3,4),j.int8)
    assert align_item_presence(obs,tactical,enabled=False) is obs
    aligned=align_item_presence(obs,tactical,enabled=True)
    np.testing.assert_array_equal(aligned,align_item_presence(aligned,tactical,enabled=True))


def test_encoder_routes_real_state_and_pid_without_touching_global_mask_or_rules():
    calls=[];original=j.zeros((24,3,4));state=SimpleNamespace(tactical_crate=j.ones((3,4),j.int8))
    def legacy(s,p,danger,*,channels):
        calls.append((s,p,danger,channels));return original[:channels]
    got=aligned_encoder(legacy,enabled=True)(state,1,'hazard',channels=24)
    assert calls==[(state,1,'hazard',24)]
    np.testing.assert_array_equal(got[7],np.ones((3,4)))
