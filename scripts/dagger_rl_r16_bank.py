"""Inactive nine-run Bank curves, same common R15 evaluation input schema."""
import inspect
from scripts import dagger_rl_r4_bank as C
from scripts.dagger_rl_r16_run import B, R, read, authorize, notify
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
        ('assert len(dev)==37 and len(finals)==7', 'assert len(dev)==55 and len(finals)==10'),
        ('dev_models=37,holdout_models=7,bankP_models_per_split=14',
         'dev_models=55,holdout_models=10,bankP_models_per_split=20'),
    ]:
        assert source.count(old) == 1
        source = source.replace(old, new)
    scope = dict(vars(C), B=B, R=R, read=read, authorize=authorize, notify=notify)
    exec(compile(source, '<R16-nine-run-frozen-Bank-lane>', 'exec'), scope)
    scope['run']()


if __name__ == '__main__':
    run()
