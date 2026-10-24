# AnomalyVFM RADIO raw pixel/global Q3 — actual Dot L4 result

The predeclared pixel-mask hypothesis failed. Native23source blobs were unchanged,532state tensors loaded strictly with0missing/unexpected/mismatched keys;355357699neural parameters. One synthetic prerequisite and160real query forwards (120normal calibration+40target), errors0/invalid0, no local neural inference. Q1/Q2/gold calls0; current input copies unchanged, no new neural training. External synthetic defect/mask pretraining is disclosed, not called untrained/generalization evidence.

| Same40targets /16defective,24normal | Missed defects | False alarms | Balanced accuracy |
|---|---:|---:|---:|
| Predeclared primary: raw mask maximum |16|1|.479167|
| Matched same-forward global summary |11|0|.656250|
| Cached current complete Q3 cascade |5|4|.760417|
| Cached current first-presence stage |4|4|.791667|

Primary mask-minus-global BA delta−.177083, paired95%CI[−.302083,−.062500]; FN-rate deltaCI[.125000,.562500]. The same-forward comparison directly contradicts the claimed benefit of this local-mask policy on this explored set. Missing-cap4/4,rust10/10,bird2/2missed. Global also misses all4missing caps plus6rust and1bird. Primary versus full current cascade BA delta−.281250,CI[−.416927,−.135417]; this is a whole-policy context comparison, not an architectural causal ablation. Preserve stage distinction rather than confusing first-stage3/15with full-cascade5/4.

Posthoc failure diagnosis only: target-only raw-mask ranking AUROC.415365,global.651042. No threshold selected, no probability interpretation, no policy revised from these outcomes. Pixel values occupy overlapping ranges in defective[.2578,.3574] and normal[.2773,.3418]targets. The displayed missing-cap/normal pair uses the same fixed0..1raw-mask overlay; visually inspecting it is not a location-accuracy evaluation. Native768square resize distorts aspect ratio, BF16native outputs have quantized/tied scores; these are disclosed limitations, not reasons to rerun with new settings. This exact backbone/pooling/calibration/threshold/preprocessing setting is rejected, no tuning/expansion/retry.

Actual saved-query latency22.797946s; run wall56.593294s includes model loading and synthetic check but excludes prior weight download and transfer roundtrip. No software-failure reruns were needed for this actual model experiment. SourceZIP d8b87b7a01926ea9004b8e3658645abcf0957cfaa1ddaaf7ba9017a7dfaa15f2; resultZIP a3dfbbf9d33acaf9325756a66502b524fd2387c68c33203856dbbe0bb62a81a1 remote/localSHA match. Original160images/tensors,160full96×96float masks/records/complete attempts,80rank-policy decisions,36source files/23official blobs/5input files,532native/saved/header shape maps and all metrics/paired20000image-resample intervals independently audited on CPU. No model forward was repeated locally. Targeted5+exact-source5tests, embeddedZIP equality/AST and actual160image CPUpreflight passed.

Artifacts: anomalyvfm_audit_v1.json, audit_anomalyvfm_results.py, anomalyvfm_raw_v1/, anomalyvfm_posthoc_score_diagnostics.json, anomalyvfm_failure_diagnostic_pair.png, immutable source/result ZIPs; scripts/run_anomalyvfm_presence_dev.py, docs/anomalyvfm_presence_dev.md, notebooks/lvlm_anomalyvfm_presence_dev_colab.ipynb. dev40already explored, gold locked, positive spatialGT absent, no generalization or verified localization claim.

The whole goal remains unmet. Q2missing is a categorical decision and was not supplied by these generic anomaly scores. Next investigate the genuinely different DeepStack Qwen3-VL8B pretrained multimodal structure on Q2classification and independent Q3presence, reusing exact cached control prompts/images and retaining strict output/format gates. Do not replace a failed score threshold with a posthoc favorable one.

2026-10-17 CPU-only erratum: cached first-presence A/B positions are permuted. Earlier audit incorrectly treated B as always positive (FN3/FP15); decoded from each actual saved prompt, correct FN4/FP4, BA.791667. Fullcascade FN5/FP4 and the new model scores/primary matched contrasts are unchanged. Old audit/report copies retained as *_before_option_semantic_fix; raw/source ZIPs unchanged, zero new modelcalls. InCTRL corrected first-stage secondary delta BA−.260417,95%CI[−.406250,−.083333]; remains rejected. See presence_semantic_correction.json and regression test.
