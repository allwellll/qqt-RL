"""One conditionally admitted, bounded CPU fresh256 matrix; no training/promotion."""
import json,sys
from pathlib import Path
from scripts import bomb_threat_a3sr_fresh as A
from scripts import bomb_threat_fresh256 as F


def main():
    name=sys.argv[1]
    assert name in ('bt_slot07_c000029_u001280_holdout','bt_slot05_c000028_u001200_holdout')
    out=Path('runs/bomb_threat_v1/iter2/a3sr/round16/fresh')/name
    assert (out/'PROTOCOL.json').exists()
    A.OUT=out
    F.OUT=out
    A.main()

if __name__=='__main__':main()
