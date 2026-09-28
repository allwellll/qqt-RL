"""带结构化日志的训练子进程执行器。"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def run_logged(command: list[str], log: str | Path, environment: dict[str, str], cwd: Path) -> None:
    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"command": command, "started_unix": started}) + "\n")
        output.flush()
        result = subprocess.run(
            command, cwd=cwd, env=environment, stdout=output,
            stderr=subprocess.STDOUT, check=False,
        )
        output.write(json.dumps({
            "exit_code": result.returncode,
            "elapsed_seconds": time.time() - started,
        }) + "\n")
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {command}")
