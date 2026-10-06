# Strict serial validation 2026-10-06

## Bot strategy gate

- Fixed `1c913b7` reference, fixed seeds, 32 seed pairs with both seat orders, 64 games per arm, tactical and real JS Hunter quick screens.
- Tactical matched result: baseline and candidate both 33 wins / 31 draws; 64/64 matched games identical, `deltaNet=0`, confidence interval `[0, 0]`.
- Hunter matched result: baseline 15 wins / 19 losses / 30 draws; candidate 16 wins / 18 losses / 30 draws; `deltaNet=0.03125`, bootstrap interval `[-0.0625, 0.125]`.
- Release gate: `false`. There is no clear improvement, so the five strategy changes were withdrawn. Candidate files and hashes remain under `runs/coop_win_20261006/recovered_candidate/`; no Bot strategy was published.
- Existing Bot browser evidence: `runs/coop_win_20261006/browser/checks.json` (desktop/mobile, no remote writes).

## A: half-tile explosion exemption

- Native collision now expands across a grid line only when the center is within `0.10 tile`; 9% and 10% remain exempt, 11% receives the normal hit.
- Coverage includes horizontal/vertical four-way symmetry, four corners, same-tick aggregation, spawn protection, chain behavior, snapshot/replay, and real browser checks.
- Commands: `node web/test_half_tile_explosion.js`, `npm test`, `node scripts/verify_half_tile_browser.js`.
- Evidence: `runs/half_tile_20261006/browser/checks.json`, `1440-half-tile.png`, `390-half-tile.png`.

## B: banana wall-corner slide

- Forced banana slide performs the minimum perpendicular alignment at an open corner, then continues in its original slide direction. Closed faces and walls/bombs introduced during the slide stop safely.
- Coverage includes four directions, both sides, closed walls, dynamic obstacles, and snapshot/replay continuation.
- Commands: `node web/test_wall_exit.js`, `npm test`, `node scripts/verify_banana_corner_browser.js`.
- Evidence: `runs/banana_corner_20261006/browser/checks.json`, `1440-banana-corner.png`, `390-banana-corner.png`.

## C: settlement overlay

- Settlement is now a transparent, full-stage overlay above the terminal canvas. It keeps result, duration, rank/total/percentile, R/restart behavior, and text-only rendering for untrusted status strings.
- Desktop `1440x1000` and mobile `390x844` Chromium checks covered win/loss/draw, rank text, non-empty terminal pixels, overlay geometry, focused-input R protection, safe restart, XSS textContent behavior, no page errors, and no horizontal overflow.
- Commands: `node scripts/verify_settlement_overlay_browser.js`, `npm test`.
- Evidence: `runs/settlement_overlay_20261006/browser/checks.json` and six outcome screenshots.

## Final verification

- `git diff --check`: passed.
- Full `npm test`: passed.
- Browser runtime used the local server at `http://127.0.0.1:8080/`; no trusted IP or ranking data was fabricated outside the mocked browser response, and no remote writes were made by the verification scripts.
