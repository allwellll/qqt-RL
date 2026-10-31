"""Restart-only adapter around frozen workers; no reward or optimizer changes."""
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

BASE = Path('runs/bomb_threat_v1')
RECOVERY = BASE / 'reboot_resume_20261030'


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def adapted_source(module, stage, receipt):
    if module == 'scripts.bomb_threat_long_train':
        source = Path(module.replace('.', '/') + '.py').read_text()
        source = replace_once(source, "    source=protocol['sources'][str(args.seed)]", """    # Resume from the retained full state, not a deleted historical source.
    retained=sorted(output.glob('u*/COMMITTED.json'))
    source=json.loads(retained[-1].read_text()) if retained else protocol['sources'][str(args.seed)]""")
        source = replace_once(source, "    smoke_end=min(total,protocol['chunk_start_update']+2)",
                              "    smoke_end=min(total,start+2)")
        # Persist a token for the latest atomic saved state at smoke end.
        source = replace_once(source, "or index==total:", "or index==total or (args.smoke and index==smoke_end):")
        source = source.replace("output/'SMOKE.json'", f"ROOT/{str(receipt / 'SMOKE.json')!r}")
        source = source.replace("output/'RESTORE.json'", f"ROOT/{str(receipt / 'RESTORE.json')!r}")
        if stage == 'bootstrap':
            hook = """    if args.verify_restore:
        token_path=output/f'NEXT_STEP_u{start:06d}.json'
        assert not token_path.exists(), 'Never overwrite an existing probe'
        token=dict(digest=probe(x['params'],st,ax,keys,selected_ring),
                   Adam_digest=digest(x['adam']),full_environment_digest=digest((st,ax,keys)),ring_digest=digest(ring),
                   provenance='post-reboot deterministic probe from SHA-verified saved state; not a pre-reboot parity proof')
        atomic_json(token_path,token)
        atomic_json(ROOT/REBOOT_RECEIPT/'BOOTSTRAP.json',dict(status='PASS',checkpoint=str(cp.relative_to(ROOT)),
            token_sha256=sha(token_path),training_updates=0,pre_reboot_nextstep_parity_available=False,unix=time.time()))
        return
"""
            source = replace_once(source, '    if args.verify_restore:\n', hook)
    else:
        original = importlib.import_module(module)
        source = original.worker_source() if hasattr(original, 'worker_source') else Path(original.__file__).read_text()
        lines = source.splitlines()
        matches = [i for i, line in enumerate(lines) if line.startswith('    end=min(total,')]
        assert len(matches) == 1
        lines[matches[0]] = '    end=min(total,start+2) if args.smoke else total'
        source = '\n'.join(lines) + '\n'
        source = source.replace("out/'SMOKE.json'", f"ROOT/{str(receipt / 'SMOKE.json')!r}")
        source = source.replace("out/'RESTORE.json'", f"ROOT/{str(receipt / 'RESTORE.json')!r}")
    return source


def main():
    protocol = json.loads(Path(sys.argv[1]).read_text())
    stage = sys.argv[4]
    del sys.argv[4:]
    module = protocol['worker_module']
    receipt = RECOVERY / f"slot{protocol['slot']:02d}"
    receipt.mkdir(parents=True, exist_ok=True)
    if stage in ('bootstrap', 'restore', 'postrestore'):
        sys.argv.append('--verify-restore')
    elif stage == 'smoke':
        sys.argv.append('--smoke')
    else:
        raise ValueError(stage)
    if module in ('scripts.bomb_threat_r1_progress_train', 'scripts.bomb_threat_r1_preserved_train'):
        from scripts.bomb_threat_r1_items_runtime import install
        install()
    source = adapted_source(module, stage, receipt)
    mod = ModuleType('reboot_frozen_worker')
    mod.__file__ = str(Path(module.replace('.', '/') + '.py').resolve())
    mod.__dict__['REBOOT_RECEIPT'] = receipt
    exec(compile(source, mod.__file__, 'exec'), mod.__dict__)
    mod.main()


if __name__ == '__main__':
    main()
