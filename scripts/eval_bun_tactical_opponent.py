#!/usr/bin/env python3
"""Evaluate a frozen actor against the host-batched 40-tick Bun rule bot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--json-out", required=True)
    parser.add_argument(
        "--per-episode", action="store_true",
        help="also store per-episode outcomes and spawn cells (enables paired comparisons)")
    parser.add_argument(
        "--host-workers", type=int, default=0,
        help="processes for rule-bot decisions + funnel labels (0 = in-process, serial)")
    args = parser.parse_args()

    from jax_bomb.bun_tactical_eval import TacticalOpponentEvaluator
    evaluator = TacticalOpponentEvaluator(host_workers=args.host_workers)
    try:
        result = evaluator.run(args.checkpoint, args.games, args.seed, args.max_steps,
                               per_episode=args.per_episode)
    finally:
        evaluator.close()
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
