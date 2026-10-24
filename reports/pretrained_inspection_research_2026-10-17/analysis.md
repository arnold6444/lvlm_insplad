# AnomalyR1 matched transfer smoke — format gate failed, no performance comparison

Actually executed in the existing Dot cloud browser / Colab L4, after the separate v1 audit-serialization repair. Frozen v2 source: `0e5e5cef9097d400fe8694b85207bfc24dc0a1171eeda9ef92cde0efede598a6`. Result ZIP: `0f4cf5b0b1dc035fb9508b95736fb12ae989824fa204811ac6fa7696d7b3f562`, observed remote and independently recovered local SHA256 match.

The predeclared smoke used five label-independent asset representatives, two independent questions each and two frozen checkpoints: **20new calls**, errors **0**. AnomalyR1:10valid/10; matched Qwen2.5-VL-3B:8valid/10, **2invalid**. Both invalid base responses started a think tag without closing it; one also omitted the final answer tag. Neither was generation-length truncation. The frozen contract excludes the dependent whole comparison; no remaining35images were run, no invalid-row deletion, parser relaxation, answer salvage, prompt tuning or invalid-response retries.

This is **an output-contract failure**, not evidence that the pretrained inspection weights do or do not improve InsPLAD accuracy. No F1/FN/FP effect or confidence interval is reported from the incomplete matched comparison. Do not select the fully valid AnomalyR1 arm alone after observing the control's format failure.

Independent CPU audit verifies19source files, all5input copies and40fixed Q1records, ten original context image byte hashes/preprocessed RGB, ten matched question/prompt/RGB/tensor hash pairs, every raw output and generated token-count receipt, all20start/completion records, strict weight audits and four publisher-LFS weight-file hashes. Both models have3754622976parameters and BF16weights. Q1calls0, gold0, no InsPLAD neural training. Processor/template/image-budget are shared and pinned. Neural logits and tokenizer decoding were not recomputed locally; exact released training-example provenance remains unaudited.

Generation tokens: base **991**, AnomalyR1 **878**, total **1869**. Summed generation latency: base **63.423s**, AnomalyR1 **59.503s**, total **122.927s**. Model load/download/hash audit times: **36.061s /81.736s**. V2wall time including loading: **263.457s**; code preparation, installation history, transfer and review excluded. V1software failure before any generation attempt took72.753s and is preserved separately in `attempt1_audit.json`/`attempt1_raw`; it is not a repeated scientific prediction.

The paper's model includes GRPO/ROAM inspection post-training on other industrial data; this inference-only experiment is not evidence of a model having received no specialized training. Explored dev is not an unseen generalization benchmark. Spatial accuracy remains unknown. Full original QA/gold/user changes are preserved.

## Next structural change

Repeated generative-format failures make another free-form reviewer a poor next use of compute. Investigate the official **InCTRL pretrained discriminative residual head**, which directly returns anomaly scores and needs no language-output parser. The public official checkpoint folder currently exposes `trained_on_visa.zip` (Drive ID `1Q7-rnGc_jkuDBqpzJDGjDO6rrAk3dWJU`, displayed2.16GB). Exact inner checkpoint/default/native loading and complete weight coverage must be established before inference. Do not rerun this AnomalyR1 prompt/model configuration or reinterpret its saved invalid responses.

Next proposed Q3 test: frozen InCTRL image score versus its patch-residual component, with normal-only calibration fixed before dev scoring, original40targets/Q1preserved and Q3FN/FP/CI measured directly. No semantic part coordinates, count difference, Q2answer transfer or language-output recovery. It cannot directly classify a missing component: Q2 remains a separate unmet endpoint. Protocol/implementation are not yet frozen or executed.

Primary sources: [AnomalyR1 official repository](https://github.com/chaoyuhao/AnomalyR1), [paper](https://arxiv.org/abs/2504.11914), [InCTRL official implementation](https://github.com/mala-lab/InCTRL).
