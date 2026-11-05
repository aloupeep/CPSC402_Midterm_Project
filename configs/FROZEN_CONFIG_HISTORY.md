# frozen_config.json amendment history
| time (UTC) | sha256 | change | evaluation records affected |
|---|---|---|---|
| 21:21 | 361d377cfc92e981176c99edeb8dba00227d89d6b39550aa28e1a22bdbb44f5e | initial freeze after development tuning (nb=512, GMRES m=40/tol=1e-4/budget=480, max_outer=30, tau_mult=100, baseline denominators, repetition policy) | E2 started under this version; solver parameters unchanged afterwards |
| 21:43 | f6e3b756345aa5ba (prefix) | trimmed E4 seeds/sizes/host repetitions for the 4-hour limit; declared before any E4 record existed | E4 plan only |
| 21:37 | 9ac7cb012396159c262692a83f668d0dd93aadd530331397fa5d01955df96698 | added the attainable-accuracy track E2a (tau_mult 0.001) as a separate diagnostic suite; no production method parameter changed | E2a only |

All evaluation/confirmation trial records carry the final hash; the solver flags used by every job are recorded verbatim in logs/jobs/*/done_*.json (cmd) and in each trial record (nb, gmres_m, gmres_tol, gmres_budget, max_outer, tau).
| 21:44 | fff8b6592f65cfcf3cab87ece4d54d7c937eab703a285b2d5dc1fbe20231dde1 | E4 seed/size plan restored to 3 seeds (n<=8192), 2 seeds (16384), 1 above after a wall-clock re-estimate; declared before any E4 record | E4 plan only |
