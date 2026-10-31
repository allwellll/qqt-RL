"""Candidate Bank protocol: common production ch7 and actual-speed P0 mask."""
from scripts import dagger_rl_r4_bank as C
from scripts.dagger_rl_r15_run import B,R,read,authorize,notify
from scripts.dagger_rl_item_observation import aligned_encoder
from scripts.dagger_rl_movement_probe import install_learner_mask
original_configure=C.T.configure
def configure():
    original_configure()
    C.E.train.make_obs=aligned_encoder(C.E.env.make_obs,enabled=True)
    install_learner_mask(C.E.train,C.E.env,enabled=True)
def run():
    C.B,C.R,C.read,C.authorize,C.notify=B,R,read,authorize,notify
    C.T.configure=configure
    C.run()
if __name__=='__main__':run()
