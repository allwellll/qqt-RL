"""Cross-process CPU stage tokens; never reserve a GPU."""
import fcntl
import time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def stage_token(base,stage,limit):
    directory=Path(base)/'cpu_tokens'/stage
    directory.mkdir(parents=True,exist_ok=True)
    handle=None
    while handle is None:
        for i in range(limit):
            candidate=(directory/f'{i:02d}.lock').open('a+')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate.close();continue
            handle=candidate;break
        if handle is None:time.sleep(.2)
    try:yield
    finally:handle.close()
