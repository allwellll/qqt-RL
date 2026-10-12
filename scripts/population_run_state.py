"""Atomic durable state and single-owner locks for population processes."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def lock(path):
    stream = Path(path).open('a+')
    try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close(); raise
    return stream


def save_resume(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp, 'wb', compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=5)
    with temp.open('rb') as stream: os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def load_resume(entry):
    if sha(entry['path']) != entry['sha256']: raise ValueError('resume SHA mismatch')
    with gzip.open(entry['path'], 'rb') as stream: return pickle.load(stream)
