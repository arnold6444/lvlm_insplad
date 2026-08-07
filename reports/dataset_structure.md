# InsPLAD fault dataset structure (manifest-verified)

## unsupervised_anomaly_detection.zip

```text
unsupervised_anomaly_detection/
├─ damper-preformed/                 train[good], test[good]
├─ damper-stockbridge/               train[good], test[good,rust]
├─ glass-insulator/                  train[good], test[good,missingcap]
├─ glass-insulator-big-shackle/      train[good], test[good,rust]
├─ glass-insulator-small-shackle/    train[good], test[good,nest]
├─ glass-insulator-tower-shackle/    train[good], test[good,rust]
├─ lightning-rod-shackle/            train[good], test[good,rust]
├─ lightning-rod-suspension/         train[good], test[good,rust]
├─ plate/                            train[good], test[good,peeling-paint]
├─ polymer-insulator/                train[good], test[good,torned-up]
├─ polymer-insulator-lower-shackle/  train[good], test[good,rust]
├─ polymer-insulator-tower-shackle/  train[good], test[good,rust]
├─ polymer-insulator-upper-shackle/  train[good], test[good,rust]
├─ spacer/                           train[good], test[good]
├─ vari-grip/                        train[good], test[good,nest,rust]
├─ yoke/                             train[good], test[good]
└─ yoke-suspension/                  train[good], test[good,rust]
```

- Asset categories: **17**
- Splits: `train`, `test`
- JPG images: **27,620**

## supervised_fault_classification.zip

```text
defect_supervised/
├─ glass-insulator/
│  ├─ train/[good, missing-cap]
│  ├─ val/[good, missing-cap]
│  └─ annotations/image_list_val.txt
├─ lightning-rod-suspension/
│  ├─ train/[good, rust]
│  ├─ val/[good, rust]
│  └─ annotations/image_list_val.txt
├─ polymer-insulator-upper-shackle/
│  ├─ train/[normal, corrosão]
│  ├─ val/[good, rust]
│  └─ annotations/image_list_val.txt
├─ vari-grip/
│  ├─ train/[good, bird-nest, rust]
│  ├─ val/[good, bird-nest, rust]
│  └─ annotations/image_list_val.txt
└─ yoke-suspension/
   ├─ train/[good, rust]
   ├─ val/[good, rust]
   └─ annotations/image_list_val.txt
```

- Asset categories: **5**
- Splits: `train`, `val`; **no test split observed**
- JPG images: **11,525**
- Annotation-like files: **5** `.txt` files named `image_list_val.txt`
- Their contents were not captured by the manifest, so their line schema is not inferred here.

## Benchmark Core intersection

1. `glass-insulator`
2. `lightning-rod-suspension`
3. `polymer-insulator-upper-shackle`
4. `vari-grip`
5. `yoke-suspension`

Anomaly detection uses the official unsupervised `test` split restricted to these five assets (**2,656 images**). Structured explainability uses supervised `val` (**6,417 images**) because the actual supervised archive has no `test` split.
