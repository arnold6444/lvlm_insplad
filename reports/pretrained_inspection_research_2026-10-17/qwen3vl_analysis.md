# Qwen3-VL8B native whole-model Q2/Q3 screen — actual Dot L4 result

Actual80independent calls, errors0/invalid0, greedyBF16/SDPA/unquantized, full750native state keys/shapes exact and finite. Same40original target IDs, exact saved control normal-reference/target bytes, prompts and option permutations. No Q1/gold/newtraining calls. Model/source pinned in docs/qwen3vl_matched_dev.md; this is a whole-model/native-tokenization replacement comparison, not proof DeepStack caused an improvement.

| Metric | Cached same-stage InternVL | Qwen3-VL |
|---|---:|---:|
| Q2correct |31/40|33/40|
| Q2macroF1 |.651546|.674064|
| Q2missing-cap correct |1/4|0/4|
| Q2rust correct |5/10|7/10|
| Q2bird-nest correct |2/2|2/2|
| Q2normal correct |23/24|24/24|
| Q3presence FN |4/16|2/16|
| Q3presence FP |4/24|5/24|
| Q3presence balanced accuracy |.791667|.833333|

Q2macroF1delta+.022518, paired95%CI[−.130026,+.170835];3repairs/1regression. The specific missing-classification target regressed and Q2will not be expanded/adopted from this result. Aggregate accuracy cannot stand in for missing improvement. Q3same-stage BA delta+.041667,CI[−.041667,+.135417];FN-rate deltaCI[−.3125,0],FP-rateCI[0,+.125]. Repairs are onebird and onerust miss; missing-cap Q3miss remains1/4. Full current cascade separately has5FN/4FP/BA.760417; it includes a position-stage failure and is not the same-stage primary baseline. No positive locationGT exists; native grid or visual-attention metadata is not location correctness.

The valid Q3screen is a weak exploratory signal, not established improvement. Before looking at any new outputs, expand **Q3presence only to the remaining200dev images**, disjoint from these40, unchanged exact cached prompts/references/options/model/native budget/generation/parser. No extra Q2/Q1/baseline reruns, no tuning. Report additional200and combined240 separately, same-stage per-defect FN/FP/BA and pairedCI, fullcascade context, all errors/invalids. Any invalid excludes the whole candidate comparison under the frozen policy. Do not call the explored240generalization. This is a necessary larger check of a valid Q3candidate; the failed Q2candidate and previously rejected anomaly-score methods are not repeated.

Generation52.576999s, full query61.227322s, two phases wall227.102791s including two model loads, excluding prior download/auth/transfers; generated160tokens, peak allocated20659534336bytes. Native17.53GBweight files remain remote, publisherLFS receipts verified remotely; local audit rehashes result/source/input/original79image bytes,750state shapes,80raw outputs/attempts/callidentities/prompt-option semantics. Q2/Q3same-image pixel tensor hashes/grids match; tensor bytes were not independently regenerated locally. No multi-hour uninterrupted-runtime guarantee.

SourceZIPa2b9eb211a06664f1acef6adc69c02182f5ca26ee8f6f6ec705afaab4453247d. ResultZIP7e6d5821ecb9ab03dd414a066be63a6e42ea0dd9660d9cc6fda45f968e8d1258, remote/local match, Drive1vF5F3l__cvpYun_9um_kvmVYkjKkP1ng. Audit qwen3vl_audit_v1.json, raw qwen3vl_raw_v1/, CPU auditor audit_qwen3vl_results.py. Current firstpresence baseline corrects earlier evaluation-only B-alwayspositive error; FN4/FP4supersedes FN3/FP15 (presence_semantic_correction.json). Existing raw inputs/outputs were preserved, no neural replay for the correction.

WholeQ2missing/Q3miss research goal remains active and unmet. Q2next work must inspect a different evidence/structure with a new testable hypothesis rather than retry this model with altered wording. Q3larger check pending.
