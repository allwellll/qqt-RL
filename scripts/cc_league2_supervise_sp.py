"""cc_league2_supervise with the high self-play trainer (scripts.cc_league2_train_sp); otherwise identical."""
from scripts import cc_league2_supervise as S

if __name__ == '__main__':
    S.TRAINER = 'scripts.cc_league2_train_sp'
    S.main()
