"""Native joint fit follows fixed COMPLETE under the same CPU owner."""
from pathlib import Path
import ast


def test_candidate_audit_runs_only_after_fixed_complete_and_cannot_fail_gpu_or_fixed_eval():
    from scripts.prestock_recipe import patched_sources
    files=['scripts/je10_ops.py','scripts/je09_stage_h5c_ops.py','scripts/je09_stage_h6_ops.py',
        'scripts/je09_route_pbrs.py','scripts/je10_worker.py','scripts/je10_evaluate.py']
    before={f:Path(f).read_text() for f in files};after=patched_sources(before)
    source=after['scripts/je10_evaluate.py'];ast.parse(source)
    assert source.index("atomic_json(out/'COMPLETE.json'")<source.index('joint_fifo_audit(BASE,g)')
    # Supplemental namespaces must not launch the default authority's audit.
    assert "if OUT.name=='authority_je10' and read(BASE/'plan.json').get('prestock_preparation'):" in source
    assert "except Exception as error:" in source
    assert "out/'DAGGER_JOINT_AUDIT_FAILURE.json'" in source
    assert patched_sources(after)==after
    assert before=={f:Path(f).read_text() for f in files}
