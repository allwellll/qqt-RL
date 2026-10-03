import numpy as np

from jax_bomb import attack_trace as at


def _grid():
    wall = np.zeros((5, 5), np.bool_)
    wall[2, 4] = True
    return wall, np.zeros((5, 5), np.bool_), np.zeros((5, 5), np.bool_)


def test_ray_cover_walls_block_without_cover_and_bombs_cover_then_stop():
    wall, bombed, brick = _grid()
    bombed[0, 2] = True
    cover = at.ray_cover(wall, bombed, brick, (2, 2), 3)
    assert cover[2, 2] and cover[2, 3] and not cover[2, 4]     # wall stops, uncovered
    assert cover[1, 2] and cover[0, 2]                         # live bomb covered
    assert cover[3, 2] and cover[4, 2] and cover[2, 0]
    assert cover.sum() == 1 + 1 + 2 + 2 + 2


def _trace(**kw):
    base = {"contact_tick": 5, "ticks": 300, "placements": [], "resolutions": [], "deaths": [], "kills": []}
    base.update(kw)
    return base


def _bomb(tick, threat, cell=(1, 1)):
    return {"tick": tick, "cell": list(cell), "enemy_cell": [1, 2], "dist": 1.0, "blast": 2, "threat": threat}


def test_classes_are_exclusive_and_trade_is_not_a_kill():
    clean = _trace(placements=[_bomb(10, True)], resolutions=[{"tick": 40, "cell": [1, 1]}],
                   deaths=[{"tick": 40, "player": 1, "cause": "by_opponent"}],
                   kills=[{"tick": 40, "surviving": True}])
    trade = _trace(placements=[_bomb(10, True)], resolutions=[{"tick": 40, "cell": [1, 1]}],
                   deaths=[{"tick": 40, "player": 0, "cause": "mutual"},
                           {"tick": 40, "player": 1, "cause": "mutual"}])
    self_kill = _trace(placements=[_bomb(10, False)], resolutions=[{"tick": 40, "cell": [1, 1]}],
                       deaths=[{"tick": 40, "player": 0, "cause": "own_bomb"}])
    threat_only = _trace(placements=[_bomb(10, True)], resolutions=[{"tick": 40, "cell": [1, 1]}])
    no_threat = _trace(placements=[_bomb(10, False)], resolutions=[{"tick": 40, "cell": [1, 1]}])
    approach = _trace()
    far = _trace(contact_tick=-1)
    labels = [at.classify(t)["class"] for t in (clean, trade, self_kill, threat_only, no_threat, approach, far)]
    assert labels == ["clean_kill", "trade", "self_kill", "threat_no_kill", "bomb_no_threat",
                      "contact_no_bomb", "no_contact"]
    c = at.classify(clean)
    assert c["funnel"] == {"contact": True, "bomb": True, "pressure_bomb": False, "pressure_safe_resolved": False,
                           "threat_bomb": True, "threat_safe_resolved": True,
                           "threat_converted": True, "surviving_kill": True}
    assert list(c["funnel"]) == list(at.FUNNEL)
    t = at.classify(trade)
    assert not t["funnel"]["surviving_kill"] and not t["funnel"]["threat_safe_resolved"]
    assert at.classify(self_kill)["unsafe_bombs"] == 1


def test_kill_then_died_and_unresolved_bomb():
    trace = _trace(placements=[_bomb(10, True), _bomb(200, False, (3, 3))],
                   resolutions=[{"tick": 40, "cell": [1, 1]}],
                   deaths=[{"tick": 40, "player": 1, "cause": "by_opponent"},
                           {"tick": 210, "player": 0, "cause": "by_opponent"}],
                   kills=[{"tick": 40, "surviving": True}])
    c = at.classify(trace)
    assert c["class"] == "kill_then_died"
    assert c["unsafe_bombs"] == 1      # second bomb never resolved before episode end and actor died


def test_summary_uses_fixed_episode_denominator():
    traces = [_trace(), _trace(contact_tick=-1), _trace(placements=[_bomb(10, True)],
                                                        resolutions=[{"tick": 40, "cell": [1, 1]}])]
    s = at.summarize(traces)
    assert s["games"] == 3
    assert s["funnel"]["contact"]["count"] == 2
    assert s["funnel"]["threat_bomb"]["rate"] == 1 / 3
    assert sum(v["count"] for v in s["classes"].values()) == 3
    assert s["threat_fraction_of_bombs"] == 1.0


def test_tracer_records_threat_resolution_and_causes():
    wall, _, brick = _grid()
    tr = at.EpisodeTracer()
    fuse0 = np.zeros((5, 5), np.int32)
    fuse1 = fuse0.copy()
    fuse1[2, 1] = 30
    no = np.zeros(2, np.bool_)
    pos = np.array([[2.5, 1.5], [2.5, 3.2]])
    tr.step(0, pos, [True, True], wall, fuse0, brick, 2, True, fuse1, no, no, no, False, False, False)
    tr.step(1, pos, [True, True], wall, fuse1, brick, 2, False, fuse0, np.array([False, True]), no,
            np.array([False, True]), False, True, True)
    out = tr.to_json()
    assert out["contact_tick"] == 0
    assert out["placements"][0]["threat"] is True
    assert out["resolutions"] == [{"tick": 1, "cell": [2, 1]}]
    assert out["deaths"] == [{"tick": 1, "player": 1, "cause": "by_opponent"}]
    assert out["kills"] == [{"tick": 1, "surviving": True}]


def test_pressure_only_counts_enemy_in_cover_during_last_fuse_window():
    wall, _, brick = _grid()
    no = np.zeros(2, np.bool_)
    fuse_empty = np.zeros((5, 5), np.int32)
    placed = fuse_empty.copy()
    placed[2, 1] = 30
    tr = at.EpisodeTracer()
    far = np.array([[2.5, 1.5], [0.5, 3.5]])          # enemy outside cross at placement
    tr.step(0, far, [True, True], wall, fuse_empty, brick, 2, True, placed, no, no, no, False, False, False)
    early = placed.copy()
    early[2, 1] = 15                                     # enemy in cover but fuse > window
    tr.step(1, np.array([[0.5, 0.5], [2.5, 2.5]]), [True, True], wall, early, brick, 2, False, early,
            no, no, no, False, False, False)
    assert tr.placements[0]["pressure"] is False
    late = placed.copy()
    late[2, 1] = at.PRESSURE_WINDOW
    tr.step(2, np.array([[0.5, 0.5], [2.5, 2.5]]), [True, True], wall, late, brick, 2, False, late,
            no, no, no, False, False, False)
    out = tr.to_json()
    assert out["placements"][0]["threat"] is False and out["placements"][0]["pressure"] is True
    c = at.classify({**out, "ticks": 3})
    assert c["funnel"]["pressure_bomb"] and c["pressure_bombs"] == 1
