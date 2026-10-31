"""Run one bounded CPU horizon extension for the registered A diagnostic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import expert_validation_diagnostics as diagnostic
from scripts.population_run_state import atomic_json, sha


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in diagnostic.CHECKPOINTS:
        raise SystemExit("usage: expert_validation_horizon.py <parent|r2_u256> <output>")
    name, output = args
    diagnostic.T.configure()
    path = diagnostic.CHECKPOINTS[name]
    params, meta = diagnostic.load_params(path)
    names, arrays = diagnostic.rollout(params, steps=128)
    result = diagnostic.summarize(name, meta, names, arrays)
    payload = dict(
        schema="qqt.expert-validation-reward-alignment-horizon/v1",
        checkpoint=name,
        registered_seed=diagnostic.SEED,
        horizon=128,
        same_initial_keys_and_env_rng=True,
        teacher_off=True,
        no_replay_to_training=True,
        result=result,
        source_sha256={
            "scripts/expert_validation_diagnostics.py": sha(
                diagnostic.R / "scripts/expert_validation_diagnostics.py"),
            "scripts/expert_validation_horizon.py": sha(Path(__file__).resolve()),
        },
        does_not_control_gpu=True,
    )
    atomic_json(Path(output), payload)


if __name__ == "__main__":
    main()
