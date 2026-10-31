"""Unique CPU/JS lane, unchanged R16 evaluation protocol for three new models."""
from pathlib import Path

source = Path(__file__).with_name('dagger_rl_r15_cpu.py').read_text()
for old, new in [
    ('scripts.dagger_rl_r15_run', 'scripts.dagger_rl_r17_run'),
    ('scripts/dagger_rl_r15_cpu.py', 'scripts/dagger_rl_r17_cpu.py'),
    ('len(finals)==6', 'len(finals)==3'),
    ('final_models=7,midpoint_models=6', 'final_models=4,midpoint_models=3'),
    ('6最终模型', '3新处理模型'),
]:
    assert old in source, old
    source = source.replace(old, new)
source = source.replace('R15', 'R17')
exec(compile(source, '<R17-fixed-CPU-lane>', 'exec'), globals())
