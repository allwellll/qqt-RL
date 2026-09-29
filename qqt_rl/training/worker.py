"""训练常驻 worker 客户端，限定可执行阶段并隔离日志。"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path


WORKER_OPERATIONS = {
    "generate_bun_tactical_v2_critic_data.py": "counterfactual",
    "train_bun_critic.py": "critic_calibration",
    "train_bun_separate_ac.py": "joint_update",
}
ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES", "JAX_PLATFORM_NAME", "JAXBOMB_RULE",
    "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "PYTHONPATH",
    "BUN_TACTICAL_BOT_PATH", "BUN_TACTICAL_BOT_EXPECTED_SHA256",
    "BUN_TACTICAL_FAMILY_JSON",
    "BUN_OPPONENT_BOT_ID", "BUN_OPPONENT_BOT_CONFIG_JSON",
}


class PersistentWorkerClient:
    def __init__(self, python: str, worker_script: str | Path,
                 cwd: str | Path, environment: dict[str, str]):
        self.process = subprocess.Popen(
            [python, "-u", str(worker_script)], cwd=Path(cwd), env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        reply = self.request("ping")
        if not reply.get("ok"):
            raise RuntimeError(f"training worker failed to start: {reply}")

    def request(self, operation: str, **payload):
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("training worker pipes unavailable")
        request_id = uuid.uuid4().hex
        self.process.stdin.write(json.dumps({
            "request_id": request_id, "operation": operation, **payload}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"training worker exited unexpectedly: {stderr}")
        reply = json.loads(line)
        if reply.get("request_id") != request_id:
            raise RuntimeError("training worker response id mismatch")
        return reply

    def run_command(self, command: list[str], log: str | Path,
                    environment: dict[str, str]) -> bool:
        script_index = 2 if len(command) > 2 and command[1] == "-u" else 1
        operation = WORKER_OPERATIONS.get(Path(command[script_index]).name)
        if operation is None:
            return False
        selected_environment = {
            key: value for key, value in environment.items()
            if key in ENVIRONMENT_KEYS}
        reply = self.request(
            operation, argv=command[script_index + 1:], log=str(log),
            environment=selected_environment)
        if not reply.get("ok"):
            raise RuntimeError(
                f"persistent worker stage failed: {operation}: {reply.get('error')}")
        return True

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


class PersistentWorkerPool:
    """每阶段一个常驻进程；进程内复用 kernel，阶段间使用独立持久缓存。"""

    def __init__(self, python: str, worker_script: str | Path,
                 cwd: str | Path, environment: dict[str, str],
                 cache_root: str | Path):
        self.python = python
        self.worker_script = Path(worker_script)
        self.cwd = Path(cwd)
        self.environment = dict(environment)
        self.cache_root = Path(cache_root)
        self.clients: dict[str, PersistentWorkerClient] = {}

    def run_command(self, command: list[str], log: str | Path,
                    environment: dict[str, str]) -> bool:
        script_index = 2 if len(command) > 2 and command[1] == "-u" else 1
        operation = WORKER_OPERATIONS.get(Path(command[script_index]).name)
        if operation is None:
            return False
        if operation not in self.clients:
            worker_environment = dict(self.environment)
            stage_cache = (self.cache_root / operation).resolve()
            stage_cache.mkdir(parents=True, exist_ok=True)
            worker_environment["JAX_COMPILATION_CACHE_DIR"] = str(stage_cache)
            self.clients[operation] = PersistentWorkerClient(
                self.python, self.worker_script, self.cwd, worker_environment)
        return self.clients[operation].run_command(command, log, environment)

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
