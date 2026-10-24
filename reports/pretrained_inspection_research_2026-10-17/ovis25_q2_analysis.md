# Ovis2.5 Q2 design40: actual complete, missing-component hypothesis rejected

Same Dot Colab L4, official native Ovis2.5-9B @ d73b2283ae2a930b7762f8d7b8b8a3f0f3b5c3bd; unquantizedBF16/eager/nonthinking, identical cached Q2 text/options and79original images. Full840native loaded keys/shapes/finite state and zero loading discrepancies, 4publisherLFS+14source/processor receipts verified.40realQ2calls +2syntheticgenerations, errors0/invalid0; Q1/Q3/gold/newInsPLADtraining0. Native new-token-only output contract preserved; strict parser unchanged.

| Q2 endpoint | cached control | Ovis2.5 |
|---|---:|---:|
| Correct /40 |31|30|
| Macro F1 |.651546|.636679|
| Missing correct /4 |1|0|
| Rust correct /10 |5|8|
| Normal correct /24 |23|20|
| Bird nest correct /2 |2|2|

Paired 20,000 resamples stratified by defect class: F1 delta -.014867,95%CI[-.181091,.148291];4repairs/5regressions. All4missing cases were misclassified; exact alternative labels are retained in audit rows. The primary missing-component hypothesis failed; rust improvement is not a reason to select an after-the-fact rust route. No expansion, changed prompts/pixels/thinking/parser/quantization or same-candidate retries. Q3 was not run and no Q3 performance is implied.

ResultZIPe33ed072a2a4835196f9edaad74b4667be59507e8063ad27e859b32c65c702a9 remote/localmatch; sourceZIP6605e353e1a559ae120f515497e01f727ea4260e760a3d1024e32dabe5a996fa. Independent CPU audit reconstructs source/input/call identities, all40raw/attempt pairs, actual option semantics, originals/EXIF/RGB, and every real pixel tensor byte from PIL+NumPy. Synthetic processor golden bytes also match. Token tensor bytes/tokenizer execution were not independently regenerated locally. Existing Q1 answer bytes preserved, gold remains locked.

Real generation39.543899s, query43.316306s, two-phase wall267.693617s including model loading/checks; excludes prior downloads/authentication/transfer. Two separate model initializations each retain a synthetic generation, total3.587510s; peak CUDA20,657,571,840bytes. A successful four-minute calculation is not a long-session reliability guarantee. Source/runtime meta inventory failurev1 and CPUimport-fixv2 are preserved separately; neither used real images or loaded weights.

Whole pretrained model comparison on already explored design40, not causal proof about tokenizers or generalization. Positive spatialGT absent, spatialaccuracyunknown. The full Q2/Q3 goal remains ACTIVE and unmet; this candidate alone is rejected. Follow a distinct evidence-based Q3 structure next.

Files: ovis25_q2_audit_v1.json, ovis25_q2_raw_v1/, audit_ovis25_results.py, audit_ovis25_pixels.py, original source/result ZIPs, docs/ovis25_q2_dev.md, notebooks/lvlm_ovis25_q2_dev_colab.ipynb, scripts/run_ovis25_q2_dev.py, tests/test_ovis25_q2_dev.py(8passed).
