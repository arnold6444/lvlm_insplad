# IG numerical diagnostic — preparation

User requested the next Colab and merge after native PR69 completeness warnings49/52 and worse location accuracy. This experiment diagnoses numerical behavior on four hash-selected expert-positive images; it does not tune localization or use location GT.

Implementation: additive `missing_ig_numerics.py`, checkpointed runner, sealed builder/notebook and focused tests. Reuse pinned FP32 native preprocessing, DINO/head and existing original/blur canaries. Midpoint32/128/512 + Gauss-Legendre128; paired FP32/FP64 accumulation; autograd/centered finite differences at three alphas/three epsilons, default/SDPA MATH contexts. Preserve signed maps, warnings, errors, old32 replay and map stability.

Scope: four fixed hash-selected cores, both baselines,40 checkpoint records; max6810forward/6448backward. No location GT in capsule, no training/LVLM/Q1/Q2/gold, no location/detection gain claim. FP64 accumulation does not imply a FP64 neural network. MATH context only controls SDPA implementations; attention class/config are recorded. Finite differences and completeness are numerical diagnostics, not defect labels.

Definition of Done: analytic numerical tests and genuine tiny random HF DINO backward/frozen weights, cache/resume/corruption/failure checks; exact sealed-source tests; audited real-image/input wiring with explicit synthetic outputs; success/failure export; narrow additive PR, merge and exact main content/Colab link verified. Native pretrained diagnostics remain **UNRUN** until the user runs the notebook.

Verification receipts are saved separately in this directory. No original data ZIPs or raw GPU output are committed. Existing dirty checkout and research results are preserved.

Completed local verification: initial20+additional1 tests, existing36 tests, final exact-capsule21 tests. Actual four-image wiring through explicitly synthetic40 records validates max6810/6448 counts, zero-call resume, and exact success/failure ZIP export. Source c7befa6f6be7/52files,5code cells. Neural pretrained calls0; native numerical findings UNRUN. Windows checkout-only newline changes normalized after confirming byte equivalence; only10 additive files are published.
