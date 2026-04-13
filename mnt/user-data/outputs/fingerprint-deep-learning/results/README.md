# Results

## Model 1 — General Classifier

Training: 20 epochs, all fingerprints combined.

```
346/346 ── 6s 17ms/step - accuracy: 0.9457 - loss: 0.1350
Test Accuracy: 0.95
```

| Metric | Value |
|---|---|
| Test Accuracy | **95%** |
| Equal Error Rate (EER) | 6.22% |
| Genuine Acceptance Rate (GAR) | 100.00% |
| False Acceptance Rate (FAR) | 6.22% |
| False Rejection Rate (FRR) | 6.24% |

**Interpretation:**
- 95% accuracy means the model correctly classifies 19 out of 20 fingerprints.
- GAR of 100% means every genuine fingerprint was correctly accepted.
- EER of 6.22% means at the optimal threshold, errors on both sides are equal and low.

---

## Model 2 — Per-Finger Classifiers

Each finger trained separately for up to 50 epochs with early stopping.

### Thumb
```
71/71 ── 1s 17ms/step - accuracy: 0.9082 - loss: 0.2159
Test Accuracy for thumb: 0.89
EER: 0.1713 | GAR: 0.8268 | FAR: 0.1713 | FRR: 0.1732
```

### Index Finger
```
68/68 ── 1s 18ms/step - accuracy: 0.8760 - loss: 0.2308
Test Accuracy for index: 0.88
EER: 0.1538 | GAR: 0.8471 | FAR: 0.1538 | FRR: 0.1529
```

### Middle Finger
```
69/69 ── 1s 17ms/step - accuracy: 0.9039 - loss: 0.2056
Test Accuracy for middle: 0.90
EER: 0.1532 | GAR: 0.8459 | FAR: 0.1532 | FRR: 0.1541
```

### Ring Finger ⭐ Best result
```
70/70 ── 1s 18ms/step - accuracy: 0.9027 - loss: 0.2053
Test Accuracy for ring: 0.90
EER: 0.1446 | GAR: 0.8554 | FAR: 0.1446 | FRR: 0.1446
```

### Little Finger (most challenging)
```
69/69 ── 1s 17ms/step - accuracy: 0.8848 - loss: 0.2227
Test Accuracy for little: 0.88
EER: 0.1774 | GAR: 0.8236 | FAR: 0.1774 | FRR: 0.1764
```

---

## Comparison Table

| Finger | Accuracy | EER ↓ | GAR ↑ | FAR ↓ | FRR ↓ |
|---|---|---|---|---|---|
| Thumb | 89% | 17.13% | 82.68% | 17.13% | 17.32% |
| Index | 88% | 15.38% | 84.71% | 15.38% | 15.29% |
| Middle | 90% | 15.32% | 84.59% | 15.32% | 15.41% |
| **Ring** | **90%** | **14.46%** | **85.54%** | **14.46%** | **14.46%** |
| Little | 88% | 17.74% | 82.36% | 17.74% | 17.64% |

> ↓ = lower is better &nbsp;|&nbsp; ↑ = higher is better

---

## Conclusion

- **Model 1** achieves higher overall accuracy (95%) because it trains on all 23,934 images combined.
- **Model 2** per-finger models are more specialized but train on fewer samples (~2,400 per finger), which limits performance.
- The **ring finger** model consistently performed best across all metrics.
- The **little finger** model showed the weakest results — likely due to its smaller ridge area and higher variability in partial latent prints.
- Both models show room for improvement, particularly in handling altered latent prints at lower quality levels.
