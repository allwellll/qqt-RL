# Strict serial validation 2026-10-06

## Bot strategy gate

- Fixed `1c913b7` reference, fixed seeds, 32 seed pairs with both seat orders, 64 games per arm, tactical and real JS Hunter quick screens.
- Tactical matched result: baseline and candidate both 33 wins / 31 draws; 64/64 matched games identical, `deltaNet=0`, confidence interval `[0, 0]`.
- Hunter matched result: baseline 15 wins / 19 losses / 30 draws; candidate 16 wins / 18 losses / 30 draws; `deltaNet=0.03125`, bootstrap interval `[-0.0625, 0.125]` (initial screen; the full recheck is recorded below).
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

## Recheck executed 2026-10-06

- The archived candidate behavior test was rerun and passed: `CPU matched protocol, mirrored intercept, timed rescue, opening partition and real glue delivery passed`.
- Tactical v2 rerun: baseline and candidate were both 35 wins / 29 draws / 0 losses; all 64 matched games were identical, `deltaNet=0`, bootstrap interval `[0, 0]`.
- Real JS Hunter rerun: baseline 25 wins / 16 draws / 23 losses; candidate 17 wins / 25 draws / 22 losses; matched `better=22`, `worse=20`, `same=22`, `deltaNet=-0.109375`, bootstrap interval `[-0.40625, 0.1875]`.
- This rerun uses the final A/B simulation and a two-player-per-team Hunter match (`size=2`); the initial Hunter quick screen used `size=1`. The results are separate experiments. Both rerun arms share identical simulation, seeds, actor order and spawns.
- Hunter safety also worsened: self traps 30 to 34, friendly traps 13 to 15, deaths 44 to 56. The confidence interval includes zero, so improvement is not demonstrated; the safety gate also fails.
- Complete raw evidence is saved in `tactical_recheck_20261006.json` and `hunter_recheck_20261006.json` beside this report, including hashes, all episode rows and the matched protocol.
- The candidate is therefore explicitly rejected and `web/bun_coop_hunter_bot.js` is restored to the committed baseline. The archived candidate and hashes remain under `runs/coop_win_20261006/recovered_candidate/`.
- Repeated browser commands passed for all three stages: `node scripts/verify_half_tile_browser.js`, `node scripts/verify_banana_corner_browser.js`, and `node scripts/verify_settlement_overlay_browser.js`, each at desktop and mobile viewports.
- Full `npm test` was rerun after removing the candidate from the publishing tree.

## Independent review corrections

- Independent review reproduced an overly wide numeric tolerance at `0.10005 tile`, banana alignment entering a side bubble, and overlapping canvas/DOM terminal text. All three findings were corrected before this final release.
- A now uses `0.10 + 1e-12` rather than the movement `EPS=1e-4`; all four just-over-threshold cases receive damage. The 9/10/11% cases remain passing.
- B sweeps the perpendicular correction pixel by pixel only for forced sliding, checking static collisions, bubbles and the target center cell. New tests cover both side corners in all four directions with walls/bubbles, including restored replay states. The originally reported side-bubble reproduction now remains stationary and clears sliding safely.
- C keeps the original terminal map/dimming, uses DOM for terminal result/details while visible, and preserves the original canvas result for replay/spectator paths without settlement. Browser assertions reject duplicate canvas terminal text and overlapping DOM rows. Updated desktop/mobile screenshots were inspected visually.
- Independent follow-up review reran half-tile, wall-exit, death-overlay and leaderboard tests, verified the two collision reproductions, and found no remaining blocking code issues. Final full `npm test` and all three desktop/mobile browser scripts passed after these corrections.
- Trusted IP configuration was not modified; browser rank values are isolated mocked responses, with no remote result submissions.
