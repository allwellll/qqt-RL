import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_eval_screen", ROOT / "scripts/score_eval_screen.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

N = 64


def cell(S=0, T=0, D=0, B=12.0, P=0.5):
    g = {k: np.zeros(N) for k in ("S", "T", "D", "P")}
    g["S"][:S] = 1
    g["T"][S:S + T] = 1
    g["D"][S + T:S + T + D] = 1
    g["P"][:int(round(P * N))] = 1
    g["B"] = np.full(N, B)
    g["alive"] = 1 - np.clip(g["T"] + g["D"], 0, 1)
    return g


def base():
    return {"tactical_v2": cell(28, 0, 2, 16.2, .83), "hunter_hard": cell(1, 0, 5, 12.5, .20),
            "hunter_normal": cell(0, 0, 1, 12.9, .31), "hunter_easy": cell(1, 0, 4, 14.5, .59)}


def test_baseline_composite_matches_preregistration():
    r = sc.screen(base(), base())
    assert r["C_base"] == pytest.approx(18 / 64)
    assert not r["pass"] and set(r["fails"]) == {"gain:C", "gain:S"}


def test_attack_gain_passes():
    c = base()
    c["tactical_v2"] = cell(34, 0, 2, 16.2, .83)
    assert sc.screen(base(), c)["pass"]


def test_gain_only_from_fewer_self_kills_is_rejected():
    c = {k: cell(int(v["S"].sum()), 0, 0, float(v["B"][0]), float(v["P"].mean())) for k, v in base().items()}
    r = sc.screen(base(), c)
    assert r["dC"] >= 6 / 64 and r["fails"] == ["gain:S"]


def test_web_hard_normal_route_counts_as_attack_gain():
    c = base()
    c["hunter_hard"] = cell(3, 0, 5, 12.5, .20)
    c["hunter_normal"] = cell(1, 0, 0, 12.9, .31)
    c["tactical_v2"] = cell(28, 0, 0, 16.2, .83)
    assert sc.screen(base(), c)["pass"]


@pytest.mark.parametrize("field,value,tag", [("B", 10.0, "B"), ("P", .65, "P")])
def test_collapse_in_one_cell_blocks(field, value, tag):
    c = base()
    c["tactical_v2"] = cell(40, 0, 2, value if field == "B" else 16.2, value if field == "P" else .83)
    assert f"tactical_v2:{tag}" in sc.screen(base(), c)["fails"]


def test_self_kill_collapse_blocks():
    c = base()
    c["tactical_v2"] = cell(40, 0, 2, 16.2, .83)
    c["hunter_easy"] = cell(1, 0, 11, 14.5, .59)
    assert "hunter_easy:D" in sc.screen(base(), c)["fails"]


def test_confirm_identical_is_not_pass_and_clear_gain_passes():
    assert not sc.confirm(base(), base(), n_boot=2000)["pass"]
    c = base()
    c["tactical_v2"] = cell(48, 0, 0, 16.2, .83)
    r = sc.confirm(base(), c, n_boot=2000)
    assert r["pass"] and r["dC_ci"][0] > 0


def test_confirm_rejects_unpaired():
    c = base()
    c["hunter_easy"] = {k: v[:32] for k, v in c["hunter_easy"].items()}
    with pytest.raises(ValueError):
        sc.confirm(base(), c)
