"""Three new models, identical common R16 Bank input and reward schema."""
import inspect
from scripts import dagger_rl_r4_bank as C
from scripts.dagger_rl_r17_run import B, R, read, authorize, notify
from scripts.dagger_rl_item_observation import aligned_encoder
from scripts.dagger_rl_movement_probe import install_learner_mask

ORIGINAL_CONFIGURE = C.T.configure


def configure():
    ORIGINAL_CONFIGURE()
    C.E.train.make_obs = aligned_encoder(C.E.env.make_obs, enabled=True)
    install_learner_mask(C.E.train, C.E.env, enabled=True)


def run():
    C.T.configure = configure
    source = inspect.getsource(C.run)
    for old, new in [
        ('assert len(dev)==37 and len(finals)==7', 'assert len(dev)==19 and len(finals)==4'),
        ('dev_models=37,holdout_models=7,bankP_models_per_split=14',
         'dev_models=19,holdout_models=4,bankP_models_per_split=8'),
    ]:
        assert source.count(old) == 1
        source = source.replace(old, new)
    scope = dict(vars(C), B=B, R=R, read=read, authorize=authorize, notify=notify)
    exec(compile(source, '<R17-fixed-Bank-lane>', 'exec'), scope)
    scope['run']()


if __name__ == '__main__':
    run()
