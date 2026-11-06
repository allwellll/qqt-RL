"""cc_league2_supervise with the fountain-variant trainer (scripts.cc_league2_train_ftn); otherwise identical.
usage: model_diagnostics.py <config> args=[arm, gpu, cpus, family]"""
from scripts import cc_league2_supervise as S

if __name__ == '__main__':
    S.TRAINER = 'scripts.cc_league2_train_ftn'
    S.main()
