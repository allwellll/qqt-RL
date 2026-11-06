"""Per-arm league publication for the CC GPU4-7 namespace (CC_SNAPSHOT_ROOT); opponents are local files."""
import os
from pathlib import Path

source = Path('scripts/combat_specialist_snapshots.py').read_text()
old = "BASE = Path('runs/combat_specialist_league_20261101')"
assert source.count(old) == 1
exec(compile(source.replace(old, "BASE = Path(os.environ['CC_SNAPSHOT_ROOT'])"), __file__, 'exec'), globals())
