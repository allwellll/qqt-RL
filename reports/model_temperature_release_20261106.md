# Native v2 temperature release (20261106)

Selected **G_frozen + τ0.3**. Both movement and ability heads use legal masked temperature sampling. The game seed and separate policy/bot RNG streams reproduce decisions. Native v2 rules, original items and traps, model seat P1, 2400 tick limit; no online teacher or policy assistance.

The requested G_frozen argmax reference and the actual previously deployed five-slot u9809 argmax were both evaluated. The old policy and its archived Review remain available.

## Paired browser screen

Five opponents, 64 shared seeds each, model P1 only. All current-engine matches were freshly run and replay-hash checked. Score = (wins + half draws) / games. These fixed scenarios do not estimate general human win rate.

| Model / mode | W/L/D | Score | Deliveries | Causal kills | Self deaths | Carry failures | Actionable stationary % | Longest stationary | c ticks | Fountain streak |
|---|---|---|---|---|---|---|---|---|---|---|
| L2_S05g_u019302_tau0.3 | 309/5/6 | 0.9750 | 277 | 179 | 39 | 2 | 19.47 | 129 | 3684 | 1.0719 |
| G_frozen_tau0.3 | 315/2/3 | 0.9891 | 291 | 177 | 18 | 4 | 21.32 | 94 | 6487 | 0.9688 |
| G_frozen_argmax | 305/7/8 | 0.9656 | 272 | 158 | 9 | 2 | 22.84 | 748 | 8577 | 0.9187 |
| online_u9809_argmax | 304/11/5 | 0.9578 | 276 | 159 | 20 | 12 | 12.70 | 1282 | 1600 | 0.9688 |
| L2_S05j_u019462_tau0.3 | 309/8/3 | 0.9703 | 270 | 145 | 156 | 1 | 19.18 | 66 | 3229 | 1.1187 |
| L2_S05i_u019382_tau0.3 | 306/9/5 | 0.9641 | 285 | 177 | 71 | 1 | 16.40 | 47 | 3920 | 1.0344 |

## Default opponent: cooperative hunter hard

| Model / mode | W/L/D | Score | Deliveries | Causal kills | Self deaths | Carry failures | Actionable stationary % | Longest stationary | c ticks | Fountain streak |
|---|---|---|---|---|---|---|---|---|---|---|
| L2_S05g_u019302_tau0.3 | 57/3/4 | 0.9219 | 56 | 25 | 10 | 2 | 22.47 | 22 | 1430 | 1.2500 |
| G_frozen_tau0.3 | 63/1/0 | 0.9844 | 62 | 27 | 5 | 1 | 23.41 | 25 | 1071 | 1.1094 |
| G_frozen_argmax | 61/3/0 | 0.9531 | 59 | 18 | 3 | 1 | 29.15 | 64 | 1534 | 1.0938 |
| online_u9809_argmax | 58/5/1 | 0.9141 | 54 | 29 | 11 | 1 | 12.36 | 82 | 340 | 1.1719 |
| L2_S05j_u019462_tau0.3 | 57/5/2 | 0.9062 | 58 | 21 | 42 | 0 | 24.20 | 66 | 1659 | 1.2500 |
| L2_S05i_u019382_tau0.3 | 59/4/1 | 0.9297 | 57 | 27 | 22 | 0 | 16.46 | 25 | 824 | 1.1875 |

## Independent holdout

Fresh seeds 2026623000–2026623087, 88 games per seat/mode, argmax/τ0.3/τ0.5; four frozen checkpoints and same-batch G_frozen reference, 2112 games. Held out from previous retests and temperature screens. Temperature selection still introduces multiple-comparison uncertainty.

| Model / mode | W/L/D | Score | Deliveries | Causal kills | Self deaths | Carry failures | Actionable stationary % | Longest stationary | c ticks | Fountain streak |
|---|---|---|---|---|---|---|---|---|---|---|
| G_frozen/greedy/P0 | 33/17/38 | 0.5909 | 22 | 17 | 13 | 8 | 32.30 | 2376 | 799 | 1.1136 |
| G_frozen/greedy/P1 | 17/33/38 | 0.4091 | 14 | 20 | 15 | 10 | 32.38 | 2389 | 3210 | 1.1932 |
| G_frozen/tau03/P0 | 41/20/27 | 0.6193 | 31 | 18 | 14 | 14 | 17.54 | 116 | 1331 | 1.5455 |
| G_frozen/tau03/P1 | 20/38/30 | 0.3977 | 18 | 15 | 23 | 13 | 29.62 | 210 | 3227 | 1.6136 |
| G_frozen/tau05/P0 | 40/26/22 | 0.5795 | 37 | 18 | 22 | 10 | 19.82 | 31 | 1182 | 1.5000 |
| G_frozen/tau05/P1 | 22/43/23 | 0.3807 | 26 | 33 | 23 | 15 | 25.70 | 63 | 2181 | 1.6250 |
| L2_S05g_u019302/greedy/P0 | 53/20/15 | 0.6875 | 33 | 17 | 13 | 1 | 42.01 | 2123 | 7442 | 1.5909 |
| L2_S05g_u019302/greedy/P1 | 34/34/20 | 0.5000 | 30 | 30 | 19 | 2 | 42.27 | 2347 | 9282 | 1.6932 |
| L2_S05g_u019302/tau03/P0 | 52/23/13 | 0.6648 | 44 | 30 | 16 | 3 | 24.63 | 50 | 800 | 1.7727 |
| L2_S05g_u019302/tau03/P1 | 41/36/11 | 0.5284 | 35 | 33 | 38 | 4 | 26.34 | 577 | 3251 | 1.8977 |
| L2_S05g_u019302/tau05/P0 | 48/31/9 | 0.5966 | 38 | 22 | 16 | 2 | 23.88 | 32 | 914 | 1.8409 |
| L2_S05g_u019302/tau05/P1 | 45/36/7 | 0.5511 | 45 | 41 | 31 | 3 | 26.35 | 77 | 2086 | 1.7386 |
| L2_S05j_u019462/greedy/P0 | 52/20/16 | 0.6818 | 38 | 20 | 14 | 2 | 17.81 | 2004 | 2862 | 1.6023 |
| L2_S05j_u019462/greedy/P1 | 31/32/25 | 0.4943 | 31 | 25 | 44 | 2 | 36.36 | 2011 | 15330 | 1.7273 |
| L2_S05j_u019462/tau03/P0 | 54/24/10 | 0.6705 | 46 | 29 | 28 | 5 | 20.27 | 32 | 1127 | 1.6705 |
| L2_S05j_u019462/tau03/P1 | 39/35/14 | 0.5227 | 34 | 18 | 31 | 1 | 23.82 | 139 | 5847 | 1.7273 |
| L2_S05j_u019462/tau05/P0 | 38/38/12 | 0.5000 | 31 | 33 | 26 | 8 | 20.10 | 98 | 1642 | 1.7955 |
| L2_S05j_u019462/tau05/P1 | 31/41/16 | 0.4432 | 31 | 33 | 43 | 7 | 23.71 | 27 | 4811 | 1.7727 |
| L2_S05i_u019382/greedy/P0 | 49/22/17 | 0.6534 | 25 | 14 | 24 | 4 | 28.28 | 2296 | 2713 | 1.6591 |
| L2_S05i_u019382/greedy/P1 | 34/35/19 | 0.4943 | 26 | 30 | 26 | 5 | 24.21 | 2351 | 3298 | 1.8977 |
| L2_S05i_u019382/tau03/P0 | 49/26/13 | 0.6307 | 38 | 19 | 30 | 3 | 18.10 | 162 | 619 | 1.8636 |
| L2_S05i_u019382/tau03/P1 | 44/32/12 | 0.5682 | 39 | 31 | 31 | 8 | 19.72 | 417 | 1680 | 2.0227 |
| L2_S05i_u019382/tau05/P0 | 52/27/9 | 0.6420 | 34 | 22 | 21 | 11 | 17.41 | 28 | 472 | 1.7614 |
| L2_S05i_u019382/tau05/P1 | 39/38/11 | 0.5057 | 40 | 31 | 37 | 11 | 17.62 | 41 | 1380 | 1.8636 |

## Decision and limits

paired score >= requested and actual online controls; default-coop W-L >= both; longest idle halved; fountain maintained; paired score 95% interval includes 0 => independent holdout P1 score/delivery/causal tiebreak; self-death excluded from decision

Eligibility checks: `{"G_frozen_tau0.3": {"coop_net_wins_not_lower_than_both_controls": true, "holdout_both_seats_longest_halved": true, "holdout_fountain_nondecreasing": true, "new_checkpoint_improves_health_holdout": true, "web_longest_actionable_halved": true, "web_score_not_lower_than_both_controls": true}, "L2_S05g_u019302_tau0.3": {"coop_net_wins_not_lower_than_both_controls": false, "holdout_both_seats_longest_halved": true, "holdout_fountain_nondecreasing": true, "new_checkpoint_improves_health_holdout": true, "web_longest_actionable_halved": true, "web_score_not_lower_than_both_controls": true}, "L2_S05i_u019382_tau0.3": {"coop_net_wins_not_lower_than_both_controls": false, "holdout_both_seats_longest_halved": true, "holdout_fountain_nondecreasing": true, "new_checkpoint_improves_health_holdout": true, "web_longest_actionable_halved": true, "web_score_not_lower_than_both_controls": false}, "L2_S05j_u019462_tau0.3": {"coop_net_wins_not_lower_than_both_controls": false, "holdout_both_seats_longest_halved": true, "holdout_fountain_nondecreasing": true, "new_checkpoint_improves_health_holdout": true, "web_longest_actionable_halved": true, "web_score_not_lower_than_both_controls": true}}`

Selected JSON SHA-256: `b0601e29ca06556d8fb561a1baa43dd6b723a8d6996fa72c879a3d1e31af24ce`. ONNX SHA-256: `0cca2473fb0b6af900d523a04fb65526d22f289486c4ec31fb9875cee671f480`. Checkpoint policy SHA-256: `f2e6c9b8c0fa9a8ac46ba483fca714c4b31e06d9185ef833cd02fb97c03a7d9b`. Sanitized metadata only; weights and ONNX bytes are unchanged.

The c category is a conservative certificate for safe progress under currently observed hazards, not an expert safety label. A shorter longest stationary interval does not mean every stationary tick is a mistake. Self deaths are reported and are not a selection veto. Fountain events are diagnostics, not rewards for bomb counts.

## Verification and rollback

Unit and full regression checks include masked sampling, seeded RNG, native R3/slot input compatibility, touch pointer cancellation, historical Review hashes, and game modes. Desktop/mobile human controls and model P1 observation are verified with the real WASM worker. Final Actions/Pages and online identity evidence are checked against the exact deployment head.

Immediate local choice rollback: select the retained five-slot u9809 model (argmax). Deployment rollback: normally revert the temperature release commit and push main; retain both old and new immutable Release assets. The preceding deployment is commit `69d9b4be7f29831371f824ff621c64e91e8c393b`. Historical Review files and physics archives were not migrated.
