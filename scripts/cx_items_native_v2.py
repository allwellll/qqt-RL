"""Private native extension, retaining the complete old checkpoint schema."""
from jax_bomb.je07_native_items import *
from scripts.cx_items_rules_v2 import available,release_deaths,VERSION
ITEM_RULE_VERSION=VERSION

# Retain the existing bird schedule, typed graveyard, flight and queue schema.
# Contract revision2 rechecks the complete legal predicate at landing, while
# excluding only that drop's own reserved cell.
import inspect as _inspect
from jax_bomb import je07_native_items as _legacy
_src=_inspect.getsource(_legacy.airdrop)
_edits={
    'def airdrop(core,tactical,field,s,key):':'def airdrop(core,tactical,field,s,key,loose=None):',
    'free=~core.wall[r,c]&~core.brick[r,c]&~core.pushable[r,c]&(core.crate[r,c]==0)&(tactical[r,c]==0)&(field[r,c]==0)&(core.fuse[r,c]==0)':
        'free=available(core,tactical,field,s._replace(fall_type=s.fall_type.at[i].set(-1)),loose)[r,c]',
    'free=available(core,tactical,field,s)&':'free=available(core,tactical,field,s,loose)&',
}
for _a,_b in _edits.items():
    assert _src.count(_a)==1,_a
    _src=_src.replace(_a,_b)
exec(compile(_src,__file__,'exec'),globals())
