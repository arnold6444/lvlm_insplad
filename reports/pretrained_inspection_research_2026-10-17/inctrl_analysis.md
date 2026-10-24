# InCTRL direct Q3 presence — actual Dot L4 run and independent audit

The fixed setting is rejected. It completed160 real query forwards (120held-out normal calibration+40targets), two synthetic observer-equivalence forwards,0model errors,0invalid numeric outputs. No Q1/Q2/gold generation or new neural training. Native checkpoint loaded all345state entries strictly,0missing/unexpected/mismatched;208712260neural parameters. The observer's outputs were bitwise identical to unwrapped native outputs on the synthetic prerequisite.

| Policy on identical40target IDs | Missed defects /16 | Normal false alarms /24 | Balanced accuracy |
|---|---:|---:|---:|
| Native holistic+patch final (primary) |15|0|.531250|
| Matched same-forward patch maximum |11|3|.593750|
| Cached first-presence stage (predeclared secondary) |4|4|.791667|
| Cached complete current Q3 cascade (additional posthoc context) |5|4|.760417|

Native final versus patch BA delta−.062500, paired95%CI[−.187500,.062500]; missed-defect-rate change CI[.0625,.5000]. The holistic head does not establish improvement and misses four additional defective targets. Missing-cap4/4missed and rust10/10missed; only1/2bird nests detected. Patch misses missing4/4,rust7/10,bird0/2. Do not call0normal false alarms a success when15/16defects were missed.

The full current cascade has a later position stage, so its saved5FN/4FP differs from its actual first-stage3FN/15FP. The earlier handoff's4FN/4FP first-stage shorthand does not describe this saved input. No cache was replaced or new baseline generated. The final-versus-full-cascade comparison is explicitly posthoc: BA delta−.229167,CI[−.385417,−.052083]. These are complete-policy comparisons with different normal reference contexts, not causal head ablations. Q1/Q2current inputs are unchanged.

Failure evidence: all five assets' normal calibration sets contain holistic scores near1(.9960–.9999999 maxima). At the frozen normal-rank rule, their final score maxima exclude almost every defective target. This is an observed overlap; the experiment does not isolate whether pretrained-domain shift, viewpoints, preprocessing or another factor caused it. No threshold, padding, layer, normal-example, checkpoint-shot or resolution tuning/expansion is justified here. No new spatial-accuracy or generalization claim.

Real-query scalar latency sum75.219748s; run wall103.299016s includes model loading/synthetic checks but excludes dependency installation, authentication and full transfer roundtrip. The two initial software-only failures are retained separately: v1launcher log polluted the initially empty output; v2RGB identity omitted the manifest's size prefix. Both stopped before model loading/real query attempts. The fixes changed initialization/logging/hash representation only. All170actual images were byte/RGB/preprocess checked locally before the successful run; no local neural inference.

Provenance: sourceZIP ebfdb42a7e89cae870409a93d7670acbb8d208111346ba97d7859f7898599372; resultZIP d9be490578b6e5bf405438d7f565ae02591f175d5de8697bd1d9e78de5e7ed22, remote/local exact match. Recovered108source files/97official Git blobs/5inputs/170original images/160raw records+attempts. Independently recomputed EXIF+dimension-prefixed RGB, preprocessing tensors, native scalar algebra, all80policy decisions, metrics and paired20000-resample intervals. Whole native neural AST is identical; only package init/three proved-unused imports differ. Safe checkpoint SHA45186ea72fdb50e4e4fa8227f19ee3204fcc26a16ac511bda11f9b84e2ce72ee. Frozen auxiliary VisA training disclosed, contamination not independently audited. Gold untouched; location GT absent; explored dev only.

Artifacts: `inctrl_audit_v3.json`, `audit_inctrl_results.py`, `inctrl_score_ranges_v3.json`, immutable result/sourceZIPs, `inctrl_raw_v3/`, `notebooks/lvlm_inctrl_presence_dev_colab.ipynb`, `scripts/run_inctrl_presence_dev.py`, `tests/test_inctrl_presence_dev.py` (9targeted+9exact-source tests), `docs/inctrl_presence_dev.md`.

The full research remains active. Q2missing classification and a usable Q3miss reduction are unmet. The next source candidate is the distinct high-resolution pretrained synthetic-anomaly pixel decoder of AnomalyVFM, not a rerun of this InCTRL setting or a relaxed generative parser.

2026-10-17 CPU-only erratum: cached first-presence A/B positions are permuted. Earlier audit incorrectly treated B as always positive (FN3/FP15); decoded from each actual saved prompt, correct FN4/FP4, BA.791667. Fullcascade FN5/FP4 and the new model scores/primary matched contrasts are unchanged. Old audit/report copies retained as *_before_option_semantic_fix; raw/source ZIPs unchanged, zero new modelcalls. InCTRL corrected first-stage secondary delta BA−.260417,95%CI[−.406250,−.083333]; remains rejected. See presence_semantic_correction.json and regression test.
