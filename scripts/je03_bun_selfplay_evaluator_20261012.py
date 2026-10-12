"""Compatibility entry point for the maintained CPU-only Stage2 authority.

Historical runs remain sealed to their original worktree; start new runs from
this revision or use the recovery entry point against an unchanged worktree.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.je03_bun_selfplay_authority_recovery_20261012 import owner

if __name__ == '__main__':
    owner()
