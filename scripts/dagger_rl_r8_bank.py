"""Reuse the frozen Bank evaluation; bind only the isolated R8 namespace."""
from scripts import dagger_rl_r4_bank as C
from scripts.dagger_rl_r8_run import B, R, read, authorize, notify


def run():
    C.B, C.R, C.read = B, R, read
    C.authorize, C.notify = authorize, notify
    C.run()


if __name__ == '__main__':
    run()
