"""Defaults and opt-in boundaries pinned for this integration."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {'web/models.json': '3eb2455b86d19a34b55f9f57d087599f5d00fcce7566bb67fb0cfaea4a880b71', 'web/index.html': 'c0fe833bdf562b3fb404dccf08c39bdd9006b0c29a51dba049b52c4ff8d434fe', 'web/app.js': '2828ce8b0b53d06cf1890ecff906520392763ea26e344319ec91180d3d7a7196', 'web/sim.js': '7d2047b7db2025608b38afec0f8283625b6f98a3d09ac49693f2a0a44d1d485b', 'web/sim-bun-native-v2.js': 'f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc', 'web/sim-stock-window-v1.js': '2767cb4e2993383addc72d792fef8947f3ff86e1dcad4f73c8e1c04752a13f45', 'web/bun_hunter_bot.js': '5cf5e78b2c7251eb6a8995a6cf39fceaea5892e523b6141a84c25f8e903e05a1', 'web/bun_coop_hunter_bot.js': 'ee3d660b4cadb91d88293c63aac4313d9061f4c001630a87485811730c80f067'}

def test_default_runtime_and_model_catalog_are_unchanged():
    for name, expected in DEFAULTS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

def test_experimental_r4_modules_are_not_default_page_dependencies():
    for name in ("web/index.html", "web/app.js", "web/model_worker.js"):
        source = (ROOT / name).read_text()
        assert "sim-kill-arena" not in source
        assert "bun_interceptor_bot" not in source

def test_agent_instructions_are_ignored_not_published():
    import subprocess
    assert "/AGENTS.md" in (ROOT / ".gitignore").read_text().splitlines()
    assert "AGENTS.md" not in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
