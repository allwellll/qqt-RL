#!/usr/bin/env python3
"""候选训练常驻 worker 的 JSON-lines 协议入口。"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from pathlib import Path


OPERATIONS = {
    "counterfactual": "scripts.generate_bun_tactical_v2_critic_data",
    "critic_calibration": "scripts.train_bun_critic",
    "joint_update": "scripts.train_bun_separate_ac",
}


def respond(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            operation = request.get("operation")
            if operation == "ping":
                respond({"request_id": request_id, "ok": True, "pid": os.getpid()})
            elif operation in OPERATIONS:
                started = time.time()
                previous = {}
                for key, value in request.get("environment", {}).items():
                    previous[key] = os.environ.get(key)
                    os.environ[key] = value
                try:
                    log_path = Path(request["log"])
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps({
                            "worker_pid": os.getpid(), "operation": operation,
                            "started_unix": started}) + "\n")
                        output.flush()
                        with redirect_stdout(output), redirect_stderr(output):
                            module = import_module(OPERATIONS[operation])
                            try:
                                module.main(request.get("argv", []))
                            except SystemExit as error:
                                if error.code not in (None, 0):
                                    raise RuntimeError(
                                        f"stage exited with code {error.code}") from error
                    respond({"request_id": request_id, "ok": True,
                             "pid": os.getpid(),
                             "elapsed_seconds": time.time() - started})
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
            else:
                respond({"request_id": request_id, "ok": False,
                         "error": f"unsupported operation: {operation}"})
        except Exception as error:
            respond({"request_id": locals().get("request_id"), "ok": False,
                     "error": f"invalid request: {error}"})


if __name__ == "__main__":
    main()
