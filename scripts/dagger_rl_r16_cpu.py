"""Inactive nine-run CPU/production-JS lane, same frozen R15 mechanics."""
from pathlib import Path

source = Path(__file__).with_name('dagger_rl_r15_cpu.py').read_text()
for old, new in [
    ('scripts.dagger_rl_r15_run', 'scripts.dagger_rl_r16_run'),
    ('scripts/dagger_rl_r15_cpu.py', 'scripts/dagger_rl_r16_cpu.py'),
    ('len(finals)==6', 'len(finals)==9'),
    ('final_models=7,midpoint_models=6', 'final_models=10,midpoint_models=9'),
    ('6最终模型', '9最终模型'),
]:
    assert old in source, old
    source = source.replace(old, new)
source = source.replace('R15', 'R16')
exec(compile(source, '<R16-nine-run-frozen-CPU-lane>', 'exec'), globals())
