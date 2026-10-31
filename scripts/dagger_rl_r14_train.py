"""One isolated process per arm/seed: only the training ch7 encoder changes."""
import sys
from scripts import dagger_rl_r12_train as L
from scripts.dagger_rl_r14_run import B,R,read,authorize,notify
from scripts.dagger_rl_item_observation import aligned_encoder

BASE_BIND=L.bind
BASE_SETUP=L.T.setup
ARM=sys.argv[2] if len(sys.argv)>2 else 'aligned_obs'


def setup():
    items,source=BASE_SETUP()
    L.C.E.train.make_obs=aligned_encoder(L.C.E.env.make_obs,enabled=ARM=='aligned_obs')
    return items,source


def bind():
    L.B,L.R,L.read,L.authorize,L.notify=B,R,read,authorize,notify
    BASE_BIND()
    L.T.setup=setup


def materialize(name):
    import inspect
    source=inspect.getsource(getattr(L,name))
    source=source.replace('scripts.dagger_rl_r12_train','scripts.dagger_rl_r14_train').replace('R12','R14')
    source=source.replace('auxiliary_collector(8,greedy=True)','auxiliary_collector(8,greedy=False)')
    if name=='formal':
        assert source.count("for seed in pr['paired_seeds']:")==1
        source=source.replace("for seed in pr['paired_seeds']:","for seed in [int(sys.argv[3])]:")
        source=source.replace("for arm in pr['arms']:","for arm in [ARM]:")
        target=" atomic_json(B/'TRAINING_COMPLETE.json',dict(passed=True,runs=records,environment_steps=3145728,optimizer_steps=6144,old_continuous_resume=False))"
        assert source.count(target)==1
        source=source.replace(target," assert len(records)==1 and records[0]['update']==32")
    scope=dict(vars(L),B=B,R=R,read=read,authorize=authorize,notify=notify,bind=bind,ARM=ARM)
    scope['probe']=globals().get('probe',L.probe)
    exec(compile(source,'<R14-fixed-R12-training-loop>','exec'),scope)
    return scope[name]


probe=materialize('probe')
if __name__=='__main__':
    if sys.argv[1]=='formal':
        assert ARM in read(B/'PROTOCOL.json')['arms']
        assert int(sys.argv[3]) in read(B/'PROTOCOL.json')['paired_seeds']
    materialize(sys.argv[1])()
