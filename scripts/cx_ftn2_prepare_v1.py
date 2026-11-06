"""Freeze independent bounded FTN2 protocol without changing any existing protocol."""
import copy,json,os,time
from pathlib import Path
from scripts.population_run_state import atomic_json,sha
BASE=Path('runs/cc_train_20261105/training')
ROOT=BASE/'ftn2_20261106'
PROTOCOL=BASE/'protocols/CX_L2_FTN2_20261106_V1.json'
SOURCES=['scripts/cx_ftn2_train_v1.py','scripts/cx_ftn2_reward_v1.py','scripts/cx_fountain_continuation_v2.py','scripts/cx_safe_frontier_v1.py','scripts/cc_reward_import_hook.py','scripts/cx_ftn2_prepare_v1.py','scripts/cx_ftn2_switch_v1.py']
def main():
 assert not PROTOCOL.exists(),PROTOCOL
 old=BASE/'protocols/CC_LEAGUE2_L2_FTN_V2.json';p=json.loads(old.read_text());spec=copy.deepcopy(p['arms']['L2_FTN'])
 cp=BASE/'formal/L2_S05g/u019302';cm=json.loads((cp/'COMMITTED.json').read_text())
 assert sha(cp/'state.pkl')==cm['state_sha256'] and sha(cp/'policy.pt')==cm['policy_sha256']
 spec['reward_variant']='cx-ftn2-continuation-frontier-v1';spec['league2']['progression']['threshold']=1.01
 spec['league2']['source']=dict(arm='L2_S05g',checkpoint=str(cp),update=19302)
 p.update(namespace=str(ROOT),namespace_id='CX-L2-FTN2-20261106-V1',id='CX-FTN2-20261106-V1',arms={'L2_FTN2':spec},seed=2026709302,
  chunk_start_update=19302,campaign_start_update=19302,chunk_updates=160,league2_chunk_updates=160,continuation_source=cm,
  compatible_checkpoint_protocol_sha256=[],cc=dict(owner='CX',created_unix=time.time(),reason='User authorized FTN2 at committed FTN boundary; healthy source, independent namespace and seed; bounded 160 updates',
   recipe='B temporal continuation, synchronous CHAIN=0; c-gated first native-safe frontier .03; concession attack clawback; original movement/custody/items unchanged',
   cuda='reviewed cond-count overlay + only disable multi_output_fusion',successor='bounded trial; no automatic long successor',parent_protocol=str(old),parent_protocol_sha256=sha(old)))
 p['curriculum']['reward_changes']=True
 p['ftn2']=dict(unit_tests=str(ROOT/'tests/UNIT_ANTIHACK.json'),native_tests=str(ROOT/'tests/NATIVE_INTEGRATION.json'),collector_abi=str(ROOT/'tests/COLLECTOR_ABI.json'),
  continuation=dict(first=0,gap=24,coefficient='.01*min(streak-1,4)',max_paid_events=20),frontier=dict(amount=.03,max_events_per_episode=1,native_selected_survivable=True,post_and_swept_safe=True),
  no_bomb_count_reward=True,no_idle_penalty=True,synchronous_chain_payment=0,package_concession='once clawback all already paid R3 attack; void pending attack; lock positive attack until episode reset',
  fork_migration='void old attack queue, retain item queue/gross, reset temporal sequence; no Actor/Adam/env reset',restore_points=[19304,19382,19462],evaluation_points=[19382,19462],budget_count_includes_smoke=True)
 for name in SOURCES:p['source_sha256'][name]=sha(name)
 # Existing closure is audited as-is; do not silently bless upstream changes.
 mismatch={s:(h,sha(s)) for s,h in p['source_sha256'].items() if sha(s)!=h}
 assert not mismatch,mismatch
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'L2_FTN2').mkdir(exist_ok=True)
 alias=BASE/'formal/L2_FTN2';assert not alias.exists() and not alias.is_symlink()
 alias.symlink_to('../ftn2_20261106/L2_FTN2',target_is_directory=True)
 dst=BASE/'league/L2_FTN2/league_snapshots';dst.mkdir(parents=True,exist_ok=True)
 os.environ.update(CC_SNAPSHOT_ROOT=str(dst.parent),CC_LEAGUE2_ARM='L2_FTN2',CC_LEAGUE2_PEERS=json.dumps(spec['league2']['peers']))
 from scripts import cc_league2_lib as L
 oldg=json.loads((BASE/'league/L2_S05g/league_snapshots/G.json').read_text())
 rows=sorted((BASE/'formal/L2_S05g').glob('u*/COMMITTED.json'))
 ring=[L.entry(str(r.parent)) for r in rows if int(r.parent.name[1:])<=19302][-4:][::-1]
 atomic_json(dst/'G.json',dict(schema=L.SCHEMA,family='G',current=L.entry(str(cp)),ring=ring,fixed=oldg['fixed'],ring_fallback=oldg.get('ring_fallback')))
 L.load('G');L.load('C',19302)
 atomic_json(PROTOCOL,p)
 atomic_json(ROOT/'PREPARED.json',dict(status='PREPARED_NOT_STARTED',experiment_id=p['id'],protocol=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),source=str(cp),seed=p['seed'],budget=160,formal_alias=str(alias),unix=time.time()))
 print(PROTOCOL,sha(PROTOCOL),flush=True)
if __name__=='__main__':main()
