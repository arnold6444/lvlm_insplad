# PR55 refined dev240 results audit — 2026-10-14

The two downloaded exports contain identical raw predictions and answers. The initial export made 2605 calls on L4; the latest export reused2605 with no model load (gpu=null). They are one experiment, not two independent repetitions. Zero errors/invalids; Q1's 1680 rows are unchanged. Source QA, canonical Git source hashes, prompts, exact controller routes/choices, full coverage and all cohort metrics checked; four main additional-dev comparison CIs recomputed with 2000 paired bootstrap resamples. Original ZIPs/raw data preserved. Summed recorded inference latency 2277.74s (~38min) excludes install/Drive/notebook wall time.

Primary comparison: additional_dev200 (normal101/anomalous99), excluding the design40. This is development data, not a new formal holdout.

| Metric | Baseline | Previous agent | Refined Q2+Q3 review |
|---|---:|---:|---:|
| Q1 F1 | .835052 | .835052 | .835052 |
| Q2 macro-F1 | .514010 | .653941 | .697842 |
| Q2 correct/200 |123|145|159|
| Q1+Q2 correct/200 |115|135|142|
| Q2 rust correct/60 |51|43|50|
| Q3 presence balanced accuracy |.559206|.810381|.759476|
| Q3 normal false-region/101 |87|23|19|
| Q3 anomalous false-none/99 |2|15|29|
| Q1+Q2+Q3-presence joint |.345|.600|.625|

Q2 revised cues vs baseline macro-F1 delta+.183832 CI[.101085,.264562], Q1Q2 joint+.135 CI[.08,.19]. Normal false-defect38→12/101, nesting8→15/18, missing-cap1→5/21, rust51→50/60. Rust is much better than previous-agent43/60 but not better than baseline51/60. Revised Q2 vs previous-agent delta+.043902 CI[−.020283,.108516], vs fresh-old-definitions+.037574 CI[−.011536,.090172]: the extra cue benefit is not established. Fresh definitions (no prior/review framing) alone score .660269; the baseline comparison does not prove a specific wording or framing cause.

Q3 refined review vs previous agent: presence BA delta−.050905 CI[−.110616,.006777], not significant; anomalous false-none delta+.141414 CI[.076923,.214956], a statistically supported increase in misses. Normal false-region23→19/101 has CI crossing zero. Review vs always-position improves BA by only+.014851 with CI[−.014599,.043483], unconfirmed despite extra calls. The review-stage experiment should not replace the previous Q3 method on this evidence.

On the five original design40 misses, three now name a region, while two missing-cap cases remain none. Three other positives become none, so design40 still misses5/16. All three original Q2 rust-regression cases remain incorrect under revised cues. Fixing some inspected cases did not improve new-cohort Q3 recall. No exact region is scored as correct: all Q3 metrics remain image-label presence-only and location accuracy null.

Candidate next direction: retain cached Q1, investigate revised/fresh-definition Q2, and retain previous-agent Q3 as the stronger recall candidate. Combining stored revised Q2 with previous Q3 produces presence joint 0.635 on additional200; this is a posthoc recombination, not a new inference or prespecified pipeline result. No setting has been frozen, no gold run performed, no code changed or results pushed by this audit. Missing-cap remains only5/21 in Q2, and Q3 still misses15/99 even under the previous agent, so remaining limits must be reported.
