"""In-process source overlay that avoids the XLA GPU multi-output reduction fault.

XLA (jaxlib 0.11.2 and 0.12.0.1, sm_90) folds the per-env "did any brick
change" guard ``any(any(a != b, (-2,-1)))`` into one full reduction over
pred[2048,13,15], tiles it as [512,780] (each row spans 4 envs), then
multi-output-fuses it with ~13 sibling env-step elementwise outputs of shape
[2048,13,15].  The generated input_and_or_reduce_select_fusion reads 1 byte
from an invalid absolute address (CUDA_ERROR_ILLEGAL_ADDRESS).

The fix computes the cross-env predicate as an int32 count (``sum > 0``)
instead of ``any``.  Same value; the convert + add reducer cannot be merged
with the per-env or-reduction, so the per-env reduce stays one row per env
([2048,195]) and the 15-output fusion compiles to a non-faulting kernel.  All
multi-output fusion stays enabled.  (``optimization_barrier`` does not work:
XLA GPU expands it before fusion; verified in OPTIMIZED_HLO.)

The shared files are sha-pinned by the training owner's PROTOCOL.json, so
they are not edited on disk.  ``install()`` writes patched copies under
``runs/cuda_repair_claude/overlay/`` and registers them under the original
module names before anything else imports them.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

OVERLAY_DIR = Path('runs/cuda_repair_claude/overlay')

# Every lax.cond guard "any env has a board change" on the batch-level reward
# path.  Each fixed alone, XLA picks the next one for the same faulting fusion.
NOTE = '# Count, not any(): avoids an XLA GPU faulting fusion (scripts/model_cuda_overlay.py).\n'
PATCHES = {
    'scripts.je09_route_pbrs': dict(
        path='scripts/je09_route_pbrs.py',
        sha256='b0a87f1ad7ac306762dbadb856211e75e49a283d9f81bf764ae615512a08dc50',
        edits=[
            ('    stock_now = jax.lax.cond(jnp.any(dirty_now),\n',
             '    ' + NOTE + '    stock_now = jax.lax.cond(jnp.sum(dirty_now, dtype=jnp.int32) > 0,\n'),
            ('    stock_next = jax.lax.cond(jnp.any(dirty_next),\n',
             '    ' + NOTE + '    stock_next = jax.lax.cond(jnp.sum(dirty_next, dtype=jnp.int32) > 0,\n'),
            ('    temporal_safe = jax.lax.cond(jnp.any(changed), time_expanded_safe,\n',
             '    ' + NOTE + '    temporal_safe = jax.lax.cond(jnp.sum(changed, dtype=jnp.int32) > 0, time_expanded_safe,\n'),
        ]),
    'scripts.je08_bun_potential': dict(
        path='scripts/je08_bun_potential.py',
        sha256='74a3308b409db110c0fb73dbf02137892f84b07016dcbc1dfda1f3753bfa8597',
        edits=[
            ("    distance = jax.lax.cond(jnp.any(dirty), rebuild, lambda _: routing['distance'], operand=None)\n",
             '    ' + NOTE + "    distance = jax.lax.cond(jnp.sum(dirty, dtype=jnp.int32) > 0, rebuild, lambda _: routing['distance'], operand=None)\n"),
        ]),
    'scripts.bun_native_v2_custody_reward': dict(
        path='scripts/bun_native_v2_custody_reward.py',
        sha256='0d013422fe8cdb53ace03d0c7632d2f6da1420f16bfe88e430cc040626effc6f',
        edits=[
            ('    fields=jax.lax.cond(dirty.any(),',
             '    ' + NOTE + '    fields=jax.lax.cond(j.sum(dirty,dtype=j.int32)>0,'),
        ]),
    'scripts.bomb_threat_reward': dict(
        path='scripts/bomb_threat_reward.py',
        sha256='2328e3ec58efcc75c1fb3fb181bf3ca24ca1bba9bff3226f74e1742f3c395b4a',
        edits=[
            ('    improved=jax.lax.cond(removed.any(),',
             '    ' + NOTE + '    improved=jax.lax.cond(j.sum(removed.any((-2,-1)),dtype=j.int32)>0,'),
        ]),
    'scripts.r3_items_reward': dict(
        path='scripts/r3_items_reward.py',
        sha256='ca9f9370a73693a958aa740a1a61f6561d5cef98b8cc763ba86df86a778b815c',
        edits=[
            ("    oldd,newd=jax.lax.cond(topology_changed.any()|~s['road_ready'].all(),road,lambda _:(s['road_best'],s['road_best']),None)\n",
             '    ' + NOTE + "    oldd,newd=jax.lax.cond((j.sum(topology_changed,dtype=j.int32)>0)|~s['road_ready'].all(),road,lambda _:(s['road_best'],s['road_best']),None)\n"),
            ('    counter=jax.lax.cond(removed.any(),',
             '    ' + NOTE + '    counter=jax.lax.cond(j.sum(removed.any((-2,-1)),dtype=j.int32)>0,'),
        ]),
}


def patched_source(name):
    patch = PATCHES[name]
    raw = Path(patch['path']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != patch['sha256']:
        raise RuntimeError(f"{patch['path']} changed on disk ({digest}); re-derive the CUDA overlay")
    text = raw.decode()
    for old, new in patch['edits']:
        if text.count(old) != 1:
            raise RuntimeError(f"{patch['path']}: overlay anchor not unique: {old.strip()[:60]}")
        text = text.replace(old, new)
    return text


class _OverlayFinder:
    """Resolve the patched module names to the overlay files at first import.

    Lazy on purpose: importing the patched modules pulls in jax, and the target
    module must set its JAX environment (cache dir, simplified constants) first.
    """
    def __init__(self, targets):
        self.targets = targets

    def find_spec(self, name, path=None, target=None):
        if name in self.targets:
            return importlib.util.spec_from_file_location(name, self.targets[name])
        return None


def install():
    """Register patched modules under their original names. Must run before any import of them."""
    for name in PATCHES:
        if name in sys.modules:
            raise RuntimeError(f'{name} was imported before the CUDA overlay')
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    installed, targets = {}, {}
    for name, patch in PATCHES.items():
        target = OVERLAY_DIR/Path(patch['path']).name
        source = patched_source(name) + '\nCUDA_OVERLAY = True\n'
        if not target.exists() or target.read_text() != source:
            target.write_text(source)
        targets[name] = str(target.resolve())
        installed[name] = dict(original=patch['path'], original_sha256=patch['sha256'], overlay=str(target),
                               overlay_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    sys.meta_path.insert(0, _OverlayFinder(targets))
    return installed
