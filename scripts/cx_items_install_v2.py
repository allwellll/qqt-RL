"""Select sealed private sources before any engine, trainer, or JAX import."""
import importlib.util,json,os,sys
from pathlib import Path
from scripts.population_run_state import sha


class Finder:
    def __init__(self,modules):self.modules=modules
    def find_spec(self,name,path=None,target=None):
        if name in self.modules:return importlib.util.spec_from_file_location(name,self.modules[name])


def install(protocol):
    t=protocol['items_transition'];path=Path(t['accepted_rule'])
    assert sha(path)==t['accepted_rule_sha256']
    rule=json.loads(path.read_text());assert rule['status']=='PASS_ACTUAL_CANONICAL_WEB_PARITY'
    for f,h in rule['source_sha256'].items():assert sha(f)==h,f
    for f,h in protocol['source_sha256'].items():assert sha(f)==h,f
    mapping=rule['private_modules']
    for name in mapping:assert name not in sys.modules,'already imported '+name
    os.environ.update(CX_ITEMS_ACCEPTED_RULE=str(path),CX_ITEMS_BASE_SOURCE=mapping['scripts.cc_league2_train'],
        JAXBOMB_BUN_LEVELS=str((path.parent/'canonical/web/assets/maps/levels.json').resolve()))
    sys.meta_path.insert(0,Finder(mapping))
    return rule


def main():
    import runpy
    p=json.loads(Path(sys.argv[1]).read_text());install(p)
    runpy.run_module(p['items_transition']['implementation_module'],run_name='__main__')


if __name__=='__main__':main()
