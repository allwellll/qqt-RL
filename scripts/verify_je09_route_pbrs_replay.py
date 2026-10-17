"""Independent arithmetic and byte-integrity check for a route replay."""
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(trace_path):
    path = Path(trace_path)
    data = json.loads(path.read_text())
    rows = data["trajectory"]
    residuals = []
    for index, row in enumerate(rows):
        expected = 0.995 * row["phi_after"] - row["phi_before"]
        residuals.append(abs(expected - row["potential_delta"]))
        if abs(row["shaped_reward"] - 0.05 * expected) > 1e-10:
            raise ValueError(f"shaped reward mismatch at tick {row['tick']}")
        if index and rows[index - 1]["state_after"] != row["state_before"]:
            raise ValueError(f"state continuity mismatch at tick {row['tick']}")
    if max(residuals, default=0) > 1e-10:
        raise ValueError("PBRS delta mismatch")
    return {"schema": "je09.route-pbrs-replay-verification/v1", "passed": True,
            "trajectory_sha256": sha(path), "tick_count": len(rows),
            "max_discounted_delta_residual": max(residuals, default=0),
            "formula": "delta=0.995*phi_after-phi_before; shaped_reward=0.05*delta",
            "state_chain_continuous": True}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), sort_keys=True))
