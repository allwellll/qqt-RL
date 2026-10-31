"""Opt-in launch-amortized hard-distance fields; same custody/cache ABI."""
import inspect
from types import FunctionType
from scripts import bun_native_v2_custody_fast as previous

REFERENCE = previous.REFERENCE
BLOCK = previous.BLOCK

# Retain the tested compaction/cache implementation exactly, changing only
# its integer shortest-path primitive. Historical modules stay reproducible.
source = inspect.getsource(previous.sparse_fields)
source = source.replace('from scripts.je08_bun_potential import shortest',
                        'from scripts.custody_distance_candidate import shortest')
source = source.replace('shortest(blocked[safe,None],j.zeros_like(blocked[safe,None]),goals[safe])',
                        'shortest(blocked[safe,None],goals[safe])')
assert source != inspect.getsource(previous.sparse_fields)
scope = dict(previous.sparse_fields.__globals__)
exec(compile(source, __file__, 'exec'), scope)
sparse_fields = scope['sparse_fields']
fields = FunctionType(previous.fields.__code__, dict(previous.fields.__globals__, sparse_fields=sparse_fields))
potential = FunctionType(previous.potential.__code__, dict(previous.potential.__globals__, fast_fields=fields))


def install():
    previous.reference.potential = potential
