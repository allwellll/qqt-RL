"""Both arms' Bank probes use the same production-aligned observation schema."""
from scripts import dagger_rl_r4_bank as C
from scripts.dagger_rl_r14_run import B,R,read,authorize,notify
from scripts.dagger_rl_item_observation import aligned_encoder
original_configure=C.T.configure
def configure():
    original_configure()
    C.E.train.make_obs=aligned_encoder(C.E.env.make_obs,enabled=True)
def run():
    C.B,C.R,C.read,C.authorize,C.notify=B,R,read,authorize,notify
    C.T.configure=configure
    C.run()
if __name__=='__main__':run()
