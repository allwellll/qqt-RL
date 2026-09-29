import json
import subprocess
import sys
from pathlib import Path

from qqt_rl.training.worker import PersistentWorkerClient


ROOT = Path(__file__).resolve().parents[1]


def test_persistent_worker_ping_and_rejects_unknown_operation():
    process = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts/bun_training_worker.py")],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps({"request_id": "one", "operation": "ping"}) + "\n")
    process.stdin.flush()
    reply = json.loads(process.stdout.readline())
    assert reply["request_id"] == "one"
    assert reply["ok"] is True
    assert reply["pid"] == process.pid
    process.stdin.write(json.dumps({"request_id": "two", "operation": "arbitrary.import"}) + "\n")
    process.stdin.flush()
    reply = json.loads(process.stdout.readline())
    assert reply["request_id"] == "two"
    assert reply["ok"] is False
    process.stdin.close()
    process.wait(timeout=5)


def test_worker_client_reuses_pid_for_whitelisted_stage(tmp_path):
    environment = dict(__import__("os").environ)
    environment.update({"PYTHONPATH": str(ROOT), "JAXBOMB_RULE": "bun"})
    client = PersistentWorkerClient(
        sys.executable, ROOT / "scripts/bun_training_worker.py", ROOT, environment)
    try:
        before = client.request("ping")["pid"]
        handled = client.run_command([
            sys.executable, str(ROOT / "scripts/train_bun_critic.py"), "--help"
        ], tmp_path / "critic-help.log", environment)
        after = client.request("ping")["pid"]
        assert handled is True
        assert before == after
        assert "--batch-size" in (tmp_path / "critic-help.log").read_text()
    finally:
        client.close()
