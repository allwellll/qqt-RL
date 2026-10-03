"""Protocol-neutral attack trace: raw per-episode events + fixed-denominator funnel.

Both the JAX Tactical evaluator and scripts/eval_web_hunter.js emit the same raw event
schema (``attack_trace_v1``); classification lives only here so the two protocols share
one definition. Actor is player 0, opponent is player 1. Ticks are pre-step indices.

Raw episode trace::

    {"contact_tick": int|-1, "ticks": int,
     "placements": [{"tick", "cell": [r, c], "enemy_cell": [r, c], "dist", "blast", "threat"}],
     "resolutions": [{"tick", "cell": [r, c]}],
     "deaths": [{"tick", "player", "cause"}]}   # cause: own_bomb|by_opponent|mutual|other
     "kills": [{"tick", "surviving"}]}          # actor sole-source kill of the opponent

``threat``: the new bomb's ray cover (range = actor blast cap, walls stop without cover,
other live bombs/bricks are covered then stop) contains the opponent's cell at placement.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "attack_trace_v1"
CONTACT_DISTANCE = 3.0
RESOLVE_WINDOW = 3          # blast linger + same-tick bookkeeping after a bomb resolves
PRESSURE_WINDOW = 10        # opponent inside a live actor bomb's cover in its last N fuse ticks
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
FUNNEL = ("contact", "bomb", "pressure_bomb", "pressure_safe_resolved", "threat_bomb",
          "threat_safe_resolved", "threat_converted", "surviving_kill")
CLASSES = ("clean_kill", "kill_then_died", "trade", "self_kill", "killed_by_bot",
           "threat_no_kill", "bomb_no_threat", "contact_no_bomb", "no_contact")


def ray_cover(wall: np.ndarray, bombed: np.ndarray, brick: np.ndarray, cell, blast: int) -> np.ndarray:
    """Cells covered by a bomb at ``cell`` with range ``blast`` (web/sim.js ``_rays`` rule)."""
    h, w = wall.shape
    out = np.zeros((h, w), np.bool_)
    r0, c0 = int(cell[0]), int(cell[1])
    out[r0, c0] = True
    for dr, dc in DIRS:
        r, c = r0, c0
        for _ in range(int(blast)):
            r += dr
            c += dc
            if r < 0 or r >= h or c < 0 or c >= w or wall[r, c]:
                break
            out[r, c] = True
            if bombed[r, c] or brick[r, c]:
                break
    return out


class EpisodeTracer:
    """Accumulates raw events for one game from host-side pre/post step snapshots."""

    def __init__(self):
        self.contact_tick = -1
        self.ticks = 0
        self.placements, self.resolutions, self.deaths, self.kills = [], [], [], []
        self._pending: dict[tuple[int, int], int] = {}

    def step(self, tick, pos, alive, wall, fuse_before, brick, blast_cap, placed, fuse_after,
             died, own_bomb, by_opponent, mutual, kill, kill_surviving) -> None:
        pos = np.asarray(pos, np.float64)
        cells = np.floor(pos).astype(int)
        both_alive = bool(alive[0]) and bool(alive[1])
        dist = float(np.abs(pos[0] - pos[1]).sum())
        if self.contact_tick < 0 and both_alive and dist <= CONTACT_DISTANCE:
            self.contact_tick = int(tick)
        if placed:
            cell = (int(cells[0, 0]), int(cells[0, 1]))
            cover = ray_cover(wall, np.asarray(fuse_before) > 0, brick, cell, int(blast_cap))
            enemy = (int(cells[1, 0]), int(cells[1, 1]))
            self.placements.append({
                "tick": int(tick), "cell": list(cell), "enemy_cell": list(enemy),
                "dist": round(dist, 3), "blast": int(blast_cap),
                "threat": bool(alive[1]) and bool(cover[enemy]), "pressure": False})
            self._pending[cell] = len(self.placements) - 1
        if alive[1]:
            enemy = (int(cells[1, 0]), int(cells[1, 1]))
            bombed = np.asarray(fuse_before) > 0
            for cell, index in self._pending.items():
                record = self.placements[index]
                if record["pressure"] or not (0 < fuse_before[cell] <= PRESSURE_WINDOW):
                    continue
                if ray_cover(wall, bombed, brick, cell, record["blast"])[enemy]:
                    record["pressure"] = True
        for cell in sorted(self._pending):
            if fuse_after[cell] <= 0:
                self.resolutions.append({"tick": int(tick), "cell": list(cell)})
        self._pending = {cell: i for cell, i in self._pending.items() if fuse_after[cell] > 0}
        for player in (0, 1):
            if died[player]:
                if mutual:
                    cause = "mutual"
                elif own_bomb[player]:
                    cause = "own_bomb"
                elif by_opponent[player]:
                    cause = "by_opponent"
                else:
                    cause = "other"
                self.deaths.append({"tick": int(tick), "player": player, "cause": cause})
        if kill:
            self.kills.append({"tick": int(tick), "surviving": bool(kill_surviving)})
        self.ticks = int(tick) + 1

    def to_json(self) -> dict:
        return {"contact_tick": self.contact_tick, "ticks": self.ticks,
                "placements": self.placements, "resolutions": self.resolutions,
                "deaths": self.deaths, "kills": self.kills}


def _placement_outcomes(trace: dict) -> list[dict]:
    """Pair every placement with its resolution and the actor/opponent deaths around it."""
    resolutions = sorted(trace["resolutions"], key=lambda r: r["tick"])
    out = []
    for placement in trace["placements"]:
        resolve = next((r["tick"] for r in resolutions
                        if r["cell"] == placement["cell"] and r["tick"] >= placement["tick"]), None)
        end = (resolve if resolve is not None else trace["ticks"]) + RESOLVE_WINDOW
        actor_died = any(d["player"] == 0 and placement["tick"] <= d["tick"] <= end
                         for d in trace["deaths"])
        converted = resolve is not None and any(
            resolve <= k["tick"] <= end for k in trace["kills"])
        out.append({**placement, "resolve_tick": resolve,
                    "safe_resolved": resolve is not None and not actor_died,
                    "converted": converted})
    return out


def classify(trace: dict) -> dict:
    """Episode funnel flags, mutually exclusive failure class and per-placement counts."""
    placements = _placement_outcomes(trace)
    threats = [p for p in placements if p["threat"]]
    pressures = [p for p in placements if p.get("pressure")]
    killers = [p for p in placements if p["converted"]]
    actor_deaths = [d for d in trace["deaths"] if d["player"] == 0]
    surviving = [k for k in trace["kills"] if k["surviving"]]
    causes = {d["cause"] for d in actor_deaths}
    funnel = {
        "contact": trace["contact_tick"] >= 0,
        "bomb": bool(placements),
        "pressure_bomb": bool(pressures),
        "pressure_safe_resolved": any(p["safe_resolved"] for p in pressures),
        "threat_bomb": bool(threats),
        "threat_safe_resolved": any(p["safe_resolved"] for p in threats),
        "threat_converted": any(p["converted"] and p["safe_resolved"] for p in threats),
        "surviving_kill": bool(surviving),
    }
    if surviving:
        label = "kill_then_died" if actor_deaths else "clean_kill"
    elif "mutual" in causes:
        label = "trade"
    elif "own_bomb" in causes:
        label = "self_kill"
    elif "by_opponent" in causes or "other" in causes:
        label = "killed_by_bot"
    elif threats:
        label = "threat_no_kill"
    elif placements:
        label = "bomb_no_threat"
    elif funnel["contact"]:
        label = "contact_no_bomb"
    else:
        label = "no_contact"
    first = lambda items: items[0]["tick"] if items else None
    return {
        "funnel": funnel, "class": label,
        "bombs": len(placements), "threat_bombs": len(threats),
        "pressure_bombs": len(pressures),
        "pressure_safe_resolved": sum(p["safe_resolved"] for p in pressures),
        "kill_bombs_pressure": sum(bool(p.get("pressure")) for p in killers),
        "threat_safe_resolved": sum(p["safe_resolved"] for p in threats),
        "threat_converted": sum(p["converted"] and p["safe_resolved"] for p in threats),
        "unsafe_bombs": sum(not p["safe_resolved"] for p in placements),
        "kill_bombs": len(killers),
        "kill_bombs_threat": sum(p["threat"] for p in killers),
        "kill_bomb_dist": [p["dist"] for p in killers],
        "kill_bomb_fuse_ticks": [p["resolve_tick"] - p["tick"] for p in killers],
        "first_contact_tick": trace["contact_tick"] if trace["contact_tick"] >= 0 else None,
        "first_bomb_tick": first(placements), "first_threat_tick": first(threats),
        "first_kill_tick": first(surviving),
        "first_actor_death_tick": first(actor_deaths),
    }


def summarize(traces: list[dict]) -> dict:
    """Fixed denominator = number of episodes; funnel rates are episode-level."""
    games = len(traces)
    rows = [classify(t) for t in traces]
    funnel = {name: sum(r["funnel"][name] for r in rows) for name in FUNNEL}
    classes = {name: sum(r["class"] == name for r in rows) for name in CLASSES}
    totals = {k: sum(r[k] for r in rows) for k in
              ("bombs", "threat_bombs", "threat_safe_resolved", "threat_converted", "unsafe_bombs",
               "kill_bombs", "kill_bombs_threat", "pressure_bombs", "pressure_safe_resolved",
               "kill_bombs_pressure")}
    kill_dist = sorted(d for r in rows for d in r["kill_bomb_dist"])
    bomb_dist = sorted(p["dist"] for t in traces for p in t["placements"])
    median = lambda xs: xs[len(xs) // 2] if xs else None
    return {
        "schema": SCHEMA, "games": games,
        "funnel": {k: {"count": v, "rate": v / games} for k, v in funnel.items()},
        "classes": {k: {"count": v, "rate": v / games} for k, v in classes.items()},
        "per_game": {k: v / games for k, v in totals.items()},
        "threat_fraction_of_bombs": totals["threat_bombs"] / max(totals["bombs"], 1),
        "threat_safe_ratio": totals["threat_safe_resolved"] / max(totals["threat_bombs"], 1),
        "threat_conversion_ratio": totals["threat_converted"] / max(totals["threat_bombs"], 1),
        "unsafe_bomb_ratio": totals["unsafe_bombs"] / max(totals["bombs"], 1),
        "kill_bomb_threat_fraction": totals["kill_bombs_threat"] / max(totals["kill_bombs"], 1),
        "kill_bomb_pressure_fraction": totals["kill_bombs_pressure"] / max(totals["kill_bombs"], 1),
        "pressure_fraction_of_bombs": totals["pressure_bombs"] / max(totals["bombs"], 1),
        "pressure_conversion_ratio": totals["kill_bombs_pressure"] / max(totals["pressure_bombs"], 1),
        "bomb_conversion_ratio": totals["kill_bombs"] / max(totals["bombs"], 1),
        "median_bomb_dist": median(bomb_dist),
        "median_kill_bomb_dist": median(kill_dist),
        "bombs_within_2": sum(d <= 2.0 for d in bomb_dist) / max(len(bomb_dist), 1),
    }
