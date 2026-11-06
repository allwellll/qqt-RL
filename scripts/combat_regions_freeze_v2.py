"""Copy the transitive source inputs needed by this isolated experiment."""
import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

SOURCE = Path(os.environ.get('QQT_SOURCE_ROOT', '.'))
OUT = Path('runs/combat_regions_independent_20261104')


def freeze():
    pending = [
        'scripts/multi_region_g_mixed_v1.py', 'scripts/multi_region_g_train_v1.py',
        'scripts/multi_region_g_cpu_validation_v1.py',
        'scripts/combat_specialist_g_league_train.py',
        'scripts/combat_specialist_g_owner.py', 'scripts/r4_k4_export.py',
        'scripts/multi_region_continuous_audit_v1.js',
        'scripts/multi_region_eval_v1.js', 'scripts/r4_combat_eval.js',
        'scripts/multi_region_natural_audit_v1.js',
        'scripts/combat_specialist_g7_source_metrics.js',
        'web/five-item-slots-review-v1.js',
        'scripts/interceptor_primary_eval_defendcap4_v6.js',
    ]
    seen = {}; missing = set()
    while pending:
        name = pending.pop()
        if name == 'web/sim.js':
            continue
        if name in seen:
            continue
        src = SOURCE / name
        if not src.is_file():
            missing.add(name)
            continue
        data = src.read_bytes()
        seen[name] = hashlib.sha256(data).hexdigest()
        dest = Path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != data:
            archive = OUT / 'checkout_before_freeze' / name
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copyfile(dest, archive)
        shutil.copyfile(src, dest)
        if src.suffix == '.py':
            tree = ast.parse(data)
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(n.name for n in node.names)
                if isinstance(node, ast.ImportFrom):
                    parent = node.module or ''
                    if node.level:
                        parent = '.'.join(Path(name).parts[:-node.level]) + ('.' + parent if parent else '')
                    modules.append(parent)
                    modules.extend(parent + '.' + n.name for n in node.names)
            for module in modules:
                if module.split('.')[0] not in ('scripts', 'jax_bomb', 'qqt_rl'):
                    continue
                candidate = module.replace('.', '/') + '.py'
                if (SOURCE / candidate).is_file():
                    pending.append(candidate)
                init = module.split('.')[0] + '/__init__.py'
                if (SOURCE / init).is_file():
                    pending.append(init)
            # Private adapters compile source paths and embedded import strings.
            text = data.decode()
            pending.extend(re.findall(r"[\"']((?:scripts|jax_bomb|web)/[\w/.-]+\.(?:py|js))[\"']", text))
            for module in re.findall(r'\b(?:scripts|jax_bomb)\.[A-Za-z_][\w.]*', text):
                candidate = module.replace('.', '/') + '.py'
                if (SOURCE / candidate).is_file():
                    pending.append(candidate)
        elif src.suffix == '.js':
            for dep in re.findall(r"require\(['\"]([^'\"]+)['\"]\)", data.decode()):
                if not dep.startswith('.'):
                    continue
                candidate = Path(name).parent / dep
                candidate = Path(__import__('os').path.normpath(candidate))
                if not candidate.suffix:
                    candidate = candidate.with_suffix('.js')
                if str(candidate) != 'web/sim.js':
                    pending.append(str(candidate))
            pending.extend(re.findall(r"[\"']((?:scripts|web)/[\w/.-]+\.js)[\"']", data.decode()))
    (OUT / 'SOURCE_INPUTS.json').write_text(json.dumps(dict(
        status='FROZEN', source_workspace=str(SOURCE), sources=seen,
        unavailable_literal_paths=sorted(missing)), indent=2) + '\n')
    print(json.dumps(dict(files=len(seen), bytes=sum((SOURCE / f).stat().st_size for f in seen),
                          missing=sorted(missing))))


if __name__ == '__main__':
    freeze()
