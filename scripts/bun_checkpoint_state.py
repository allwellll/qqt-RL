"""Process observation and safe checkpoint boundary predicates; no signalling."""
import os
from pathlib import Path


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def boundary(phase, interval=128):
    return (phase['phase'] == 'ROLLOUT_OR_COMPILE'
            and phase['generation'] > 0
            and (phase['generation'] - 1) % interval == 0)


def cpu_command(command, affinity_variable):
    """Inherit allocated cores unless the operator explicitly sets an affinity."""
    affinity = os.environ.get(affinity_variable)
    return ['taskset', '-c', affinity, *command] if affinity else command
