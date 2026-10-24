# Q1 fixed Q2/Q3 independent dev run — 2026-10-13

Source: `dev_q2_q3_agent_results.zip` SHA256 `a235a8c4ddf856e5bde91ac1aa8bf780ced7ca5eabefe8e3bae1ad4771c98ace`. InternVL, merged commit `7edee7c758826e0b057d66fd72005eb5301a806a`, L4. Dev40: normal24/anomalous16. 232 unique calls, zero errors/invalid answers, all 160 Q1 rows unchanged. Source QA, raw choices, metrics and 2000-resample paired CIs checked offline. Inference latency sum 264.832s excludes installation/Drive/notebook wall time.

| Metric | Baseline | Q2 define + Q3 independent | Matched reask |
|---|---:|---:|---:|
| Q1 F1 | .741 | .741 | .741 |
| Q2 macro-F1 | .489 | .697 | .300 |
| Q2 correct /40 | 29 | 31 | 23 |
| Q1+Q2 correct /40 | 26 | 28 | 23 |
| Q3 presence balanced accuracy | .563 | .760 | .563 |
| Q3 normal false-region /24 | 21 | 4 | 21 |
| Q3 anomalous false-none /16 | 0 | 5 | 0 |
| Q1+Q2+Q3-presence correct /40 | 10 | 25 | 7 |

Q2 per-label correct counts: normal19→22/24, bird-nest0→2/2, missing-cap2→2/4, rust8→5/10. Five wrong→correct and three correct→wrong; rust loss includes two rust→normal and one rust→bird-nest. Q2 F1 gain vs baseline CI [-.065474,.342542] includes zero; small sample does not establish improvement. Against equal-call reask, Q2 delta+.396597 CI [.162120,.550145].

Q3 presence delta+.197917 CI [.041648,.333333]. Normal false-region delta−.708333 CI [−.88,−.50], but anomalous false-none increases+.3125 CI [.090909,.5625]. Four misses originate in the presence task (missing-cap/rust/rust/bird-nest), one missing-cap passes presence but position answers none. These are documented image-label proxy errors, not verified spatial failures. Always-none and always-region controls each score .5 balanced accuracy; agent .760 exceeds them.

Q3 location accuracy remains null. Wrong regions on anomalous images still count as presence-correct. This ten-choice auxiliary experiment does not prove original gold localization improvement or generalization to another model. Per-asset scores with no anomalous support (polymer) only assess normal specificity.

Next: keep Q1, retain Q2 definition/Q3 independent tools as candidates, investigate the rust regressions and the four presence false-negatives plus one position-stage false-none. Confirm on a larger dev cohort before selecting/fixing settings or running gold. No source code changed, inference invoked, selection written, gold read, or results pushed by this audit. Original archive and all raw files preserved in this unique hash directory.
