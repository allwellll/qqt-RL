"""Opt-in training encoder correction; imports never change a live run."""
import jax.numpy as j


def align_item_presence(obs, tactical_crate, *, enabled):
    if not enabled or obs.shape[-3] <= 7:
        return obs
    # Native ch7 sees pickup crates, including tactical pickups. Placed field
    # items remain visible only in their own channels and never enter ch7.
    return obs.at[...,7,:,:].set(j.maximum(obs[...,7,:,:],
        (tactical_crate>0).astype(obs.dtype)))


def aligned_encoder(legacy, *, enabled):
    def encode(state, pid, danger=None, channels=24):
        return align_item_presence(legacy(state, pid, danger, channels=channels),
                                   state.tactical_crate, enabled=enabled)
    return encode
