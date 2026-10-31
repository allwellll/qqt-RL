"""Same declared own-label switch-fit diagnostic; no interaction/optimization."""
from pathlib import Path

source = Path(__file__).with_name('dagger_rl_r16_diagnostics.py').read_text()
for old, new in [('scripts.dagger_rl_r16_run', 'scripts.dagger_rl_r17_run'),
                 ('expected=9', 'expected=3'), ('saved == 9', 'saved == 3'), ('runs=9', 'runs=3')]:
    assert old in source, old
    source = source.replace(old, new)
exec(compile(source, '<R17-three-fixed-fit-lane>', 'exec'), globals())
