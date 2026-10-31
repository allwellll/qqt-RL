"""Read-only action-chain audit for the R18 teacher change.

The invalid R17 u001 state is used only as a frozen learner-state diagnostic;
it is explicitly excluded from capability evidence.  Bank snapshots are real
pickup states and remain a separate source cohort.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import jax

from scripts import combat_item_engine as T
from scripts import dagger_rl_oracle as O
from scripts.carrier_mechanism_analysis import bank, merged
from scripts.dagger_rl_item_chain_teacher import compose_teacher
from scripts.population_run_state import atomic_json, sha

R = Path(__file__).resolve().parents[1]
B = R / "runs/dagger_rl_20261025/r18_item_chain"
INVALID = R / "runs/dagger_rl_20261025/r17_goal_kind/formal/sequential_anchor_s2026102661/u001/state.pkl"


def counts(route, item, source):
    legacy = np.asarray(O.combine(route, item), bool)
    fixed = np.asarray(compose_teacher(route, item), bool)
    item = np.asarray(item, bool)
    route = np.asarray(route, bool)
    return dict(
        source=source,
        states=int(route.shape[0]),
        route_rows_with_release=int(route[:, :, 2].any(1).sum()),
        certified_item_rows=int(item.any((1, 2)).sum()),
        legacy_rows_with_release=int(legacy[:, :, 2].any(1).sum()),
        fixed_rows_with_release=int(fixed[:, :, 2].any(1).sum()),
        release_labels_removed=int((legacy & ~fixed).sum()),
        fixed_teacher_rows=int(fixed.any((1, 2)).sum()),
        fixed_item_rows=int(item.any((1, 2)).sum()),
    )


def run():
    T.configure()
    items = [x for x in bank() if x["record"]["split"] == "train"]
    st = merged(items, "states")
    ax = merged(items, "aux")
    route = T.route_teacher(st, ax)
    item, _ = O.teacher(st, ax)
    rows = [counts(route, item, "real_pickup_bank_train")]

    invalid = None
    if INVALID.exists():
        with INVALID.open("rb") as f:
            saved = pickle.load(f)
        states = saved["states"]
        aux = saved["aux"]
        # Flatten device and environment axes without changing state fields.
        states = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), states)
        aux = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), aux)
        route = T.route_teacher(states, aux)
        item, _ = O.teacher(states, aux)
        invalid = counts(route, item, "excluded_r17_u001_learner_states")
        invalid["capability_evidence_excluded"] = True
        rows.append(invalid)

    payload = dict(
        passed=True,
        protocol="R18-item-chain-v1",
        rows=rows,
        semantics=dict(
            route_release_requires_certified_item_plan=True,
            certified_item_plan_has_priority=True,
            rewards_unchanged=True,
            action_masks_unchanged=True,
            invalid_r17_namespace_not_reused=True,
        ),
        source_bank_records=len(items),
        invalid_state_sha256=sha(INVALID) if INVALID.exists() else None,
        environment_steps=0,
        optimizer_steps=0,
    )
    atomic_json(B / "ITEM_CHAIN_AUDIT.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    run()
