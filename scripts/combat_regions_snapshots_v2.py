"""Separate league publication; frozen imported opponents are local files."""
from pathlib import Path

source = Path('scripts/combat_specialist_snapshots.py').read_text()
old = "BASE = Path('runs/combat_specialist_league_20261101')"
assert source.count(old) == 1
exec(compile(source.replace(old, "BASE = Path('runs/combat_regions_independent_20261104/training')"),
             __file__, 'exec'), globals())
