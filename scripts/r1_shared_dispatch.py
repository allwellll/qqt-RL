"""Allow the existing R1/R2 shared slot, rejecting every unknown compute app."""
import csv
import json
import subprocess
from pathlib import Path


def authorized_cotenants(slot, entry, r2_slots, applications):
    if slot != 7 or not entry or not entry.get('r1_items_applied'):
        return False
    partner = r2_slots.get('c', {})
    if partner.get('gpu') != slot:
        return False
    expected = (partner.get('worker_module', 'scripts.bun_native_v2_train'), partner.get('run'))
    return len(applications) == 1 and applications[0] == expected


def shared_free_slots(slots, r2_path):
    if not r2_path.exists():
        return set()
    r2_slots = json.loads(r2_path.read_text())
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    identifiers = {row[1].strip(): int(row[0]) for row in csv.reader(cards.splitlines())}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    applications = []
    for row in csv.reader(raw.splitlines()):
        if identifiers.get(row[0].strip()) != 7:
            continue
        try:
            if Path('/proc', row[1].strip(), 'cwd').resolve() != Path(__file__).resolve().parents[1]:
                return set()
            args = Path('/proc', row[1].strip(), 'cmdline').read_bytes().split(b'\0')
            index = args.index(b'-m')
            protocol = Path(args[index + 2].decode())
            if not str(protocol).startswith('runs/bun_native_v2_20261030/protocols/'):
                return set()
            applications.append((args[index + 1].decode(), args[index + 3].decode()))
        except (OSError, ValueError, IndexError, UnicodeError):
            return set()
    return {7} if authorized_cotenants(7, slots.get('7'), r2_slots, applications) else set()
