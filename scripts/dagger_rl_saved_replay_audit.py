"""Recover replay indices using the exact frozen treatment, with passive row IDs."""
import numpy as np
from scripts.dagger_rl_source_replay import aggregate_with_sources


def replay_indices(history, *, seed, update, balance_items):
    # Only the copied audit payload is tagged. Arrival IDs, source metadata and
    # family labels are preserved; no training state or random stream is changed.
    rows = history['rows']
    tagged = (np.arange(len(rows[0]),dtype=np.int64)[:,None],) + rows[1:]
    copied = dict(history,rows=tagged)
    empty = tuple(a[:0] for a in tagged)
    _, replay, stats = aggregate_with_sources(empty,np.empty(0,np.int32),copied,
        seed=seed,update=update,strategy='reservoir',balance_items=balance_items)
    return replay[0][:,0],stats
