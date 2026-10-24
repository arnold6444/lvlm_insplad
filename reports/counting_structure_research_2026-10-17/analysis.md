# CountGD++ fixed glass-part prerequisite — rejected

Existing Dot cloud browser / Colab NVIDIA L4 completed model4, and the result ZIP was recovered and independently audited. Remote and local SHA256: `715ce3623eba1ce4b435195b160fa902f05c4ba20345998bf42cf520d1a45f0b`. Source: `415af76c2baf918334f76419ce772c9c0070ee7b638d9725280da3792cd8ef10`.

The counting-pretrained official CountGD++ path differs from the earlier generic detector, but this fixed configuration does not supply sufficiently reliable glass-part evidence. Normal originals with proposals: **5/7**, required **7/7**. Flip repeatability: **2/7**, required **6/7**. Visual review: text-only **5/7**, automatic exemplars **4/7**, required automatic **6/7** without degradation. In the single-disc close crop, automatic exemplars add two metal-body boxes to one glass box. Two other normal images have zero proposals. See `qualitative_review.json` and `normal_countgd_contact.jpg`.

There were **38 new completed image forwards + 1 exactly reused completed first pass = 39 unique completed forwards**; five second passes explicitly skipped because there were no pseudo-exemplars. Model4 errors: **0**. The previous model3 adapter attempt includes **1 failed forward attempt** after that preserved first pass. Cumulative real-image forward attempts across these software attempts: **40**. The two earlier environment/strict-load stops made zero image forwards. No LVLM, Q1, gold or new neural training calls.

Independent CPU audit checks all 35,100 query boxes/token probabilities, disc-token selection, strict threshold, pseudo-exemplar geometry, original15 byte hashes, effective83 source files, exact reused raw/source provenance, counters and flip arithmetic. All named neural checkpoint parameters were present, with zero unexpected keys; no local neural recomputation was performed. The official pure-PyTorch CUDA backend passed analytic/CPU-CUDA checks; native custom-kernel equivalence was not tested.

Measured bootstrap wall time: **61.691s**, probe including model load: **49.182s**. Unique completed forward latency sum, including the reused first pass: **13.270s**. Installation/authentication/transfer/review are not included in those probe timings. This is not a long-duration reliability guarantee.

No count or point GT exists: count/spatial accuracy remain unknown. Q2/Q3 effects and confidence intervals were not measured by this prerequisite. Do not tune thresholds, rewrite captions, enlarge this configuration or connect its boxes to the LVLM. The overall research continues with a distinct evidence structure; this branch failure is not completion of the user goal.

Sources: [official CountGD++ implementation](https://github.com/niki-amini-naieni/CountGDPlusPlus), [paper](https://arxiv.org/abs/2512.23351). This adapted reduced path excludes paid enhancement and external exemplars; it is not full paper reproduction.
