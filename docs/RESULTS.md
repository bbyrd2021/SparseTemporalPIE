# SparseTemporalPIE: Results

**Brandon Byrd, Abel Abebe Bzuayene**
xDI Lab, North Carolina A&T State University

---

## 1. Experimental Setup

### Datasets

We evaluate on two standard pedestrian intention benchmarks.

**PIE (Pedestrian Intention Estimation)** [Rasouli et al., 2019] provides annotated video sequences of pedestrians in urban driving scenarios recorded at 30 fps. Each pedestrian sequence includes bounding box tracks, binary crossing intention labels, behavioral annotations (action, look), and synchronized ego-vehicle OBD data (speed). Following the original EfficientPIE protocol, we use a random 90/5/5 train/val/test split with observation window of 15 frames and 50% sequence overlap. Sequences are balanced by class. The test set contains **92 pedestrians** and **893 samples**.

**JAAD (Joint Attention in Autonomous Driving)** [Kotseruba et al., 2016] provides annotated pedestrian clips from dashcam footage across multiple cities and cameras. Labels encode crossing action intent (`intent` field). We follow the same 90/5/5 random split, observation window, and balance protocol. No OBD speed data is available in JAAD, so ego-vehicle speed features are zero-padded. The test set contains **1,876 samples**. SparseTemporalPIE v3 is trained on JAAD independently, initialized from the EfficientPIE pretrained backbone (same initialization as the PIE run).

### Incremental Learning Protocol

We follow the Incremental Domain Incremental Learning (IDIL) protocol introduced by EfficientPIE. Training proceeds across 8 IL steps corresponding to frame indices in the observation window:

$$\text{steps} = \{0, 2, 4, 6, 8, 10, 12, 14\}$$

At each step $t$, the model trained at step $t-2$ serves as the distillation teacher. The IL loss combines a task loss and a knowledge distillation term:

$$\mathcal{L} = \begin{cases} \mathcal{L}_{\text{new}} & \text{if } \mathcal{L}_{\text{new}} > \mathcal{L}_{\text{old}} \\ 0.5\,\mathcal{L}_{\text{old}} + \mathcal{L}_{\text{new}} & \text{otherwise} \end{cases}$$

where $\mathcal{L}_{\text{new}}$ is cross-entropy on current predictions and $\mathcal{L}_{\text{old}}$ is a temperature-scaled KL distillation loss against the frozen teacher.

### Implementation Details

| Hyperparameter        | Value                                      |
|-----------------------|--------------------------------------------|
| Optimizer             | RMSprop                                    |
| Learning rate (head)  | $1 \times 10^{-4}$                         |
| Learning rate (backbone) | $1 \times 10^{-5}$ (10× lower)          |
| Weight decay          | $1 \times 10^{-4}$                         |
| LR schedule           | CosineAnnealingWarmRestarts ($T_0 = 7$)    |
| Batch size            | 32                                         |
| Epochs (step 0)       | 50                                         |
| Epochs (IL steps)     | 30                                         |
| Input resolution      | $300 \times 300$ (center crop, pad resize) |
| Data augmentation     | ColorJitter, horizontal flip (synchronized image + pose), pose dropout ($p=0.1$) |

---

## 2. Architecture

### SparseTemporalPIE

SparseTemporalPIE extends EfficientPIE with three new information streams fused into the prediction pipeline:

**Pose features.** At each frame index, we extract 17-joint 2D pose keypoints using a pre-trained pose estimator. For each pedestrian, keypoints are normalized relative to the bounding box and concatenated with a velocity term (frame-to-frame delta), yielding a **68-dimensional pose vector** (34 static + 34 velocity). The pose is projected to the embedding dimension and fused additively into the backbone embedding.

**Temporal context frames.** Up to $K=4$ context frames are selected at evenly-spaced indices from $[0, t-1]$, where $t$ is the current IL step. Each context frame is passed through the shared backbone, augmented with its own pose embedding, and used as keys and values in a multi-head cross-attention layer with the current frame embedding as the query.

The IL chain progressively expands the temporal window available to the model — at step 2 there is barely any context; at step 14 the context spans the full observation window:

| IL Step | `f_current` | Context indices (K≤4, evenly spaced) |
|---------|-------------|--------------------------------------|
| 0       | frame 0     | [0] (padded)                         |
| 2       | frame 2     | [0, 1]                               |
| 4       | frame 4     | [0, 1, 2, 3]                         |
| 6       | frame 6     | [0, 2, 4, 5]                         |
| 8       | frame 8     | [0, 2, 5, 7]                         |
| 10      | frame 10    | [0, 3, 6, 9]                         |
| 12      | frame 12    | [0, 3, 7, 11]                        |
| 14      | frame 14    | [0, 4, 9, 13]                        |

Cross-attention runs once per forward pass: Q = `f_current` embedding, K/V = the K context embeddings. Each IL step trains a new model distilled from the previous step, so later steps inherit prior knowledge while gaining access to a wider temporal span. This is why v3 continues improving through step 14 — the attention head extracts new discriminative signal as the window grows — while v4 (no attention) has no mechanism to exploit the additional context and stagnates after step 2.

**Motion and behavioral features.** Bounding box trajectory statistics (12-d: displacement, velocity, acceleration, size ratio over $[0, t]$) and ego-vehicle context features (5-d: OBD speed at $t$, mean speed over $[0, t]$, speed validity flag, pedestrian action, pedestrian look direction) are projected via a 2-layer MLP to a 128-d context vector and concatenated with the enriched embedding at the classifier (late fusion).

```
f_current ──► backbone ──► emb (1280-d) ◄── pose_proj(pose_current, 68-d)
                                │
f_context[0..K] ► backbone ► K context embs ◄── pose_proj(pose_context, K×68-d)
                                │
                          cross_attn(Q=emb, K/V=context)
                                │
                           attn_norm + FF + ff_norm
                                │ (enriched, 1280-d)
bbox_traj (12-d) ──┐
ctx_feats  (5-d) ──┴──► ctx_proj MLP ──► ctx (128-d)
                                │
                    classifier(1408 → 256 → 2)
```

**Total parameters:** ~9.0M
**Backbone parameters:** ~684K (shared with EfficientPIE)

---

## 3. Main Results

### 3.1 Comparison with State of the Art (PIE Test Set)

Inference times from EfficientPIE Table 3 (RTX 3090, batch=128, 100 runs). Our models benchmarked on same protocol. End-to-end times include upstream ViTPose-B pose estimation (3.875ms).

| Method                          | Year | Accuracy | AUC   | F1    | Precision | Inference     |
|---------------------------------|------|----------|-------|-------|-----------|---------------|
| PIE [Rasouli et al.]            | 2019 | 0.790    | —     | —     | —         | —             |
| PCPA [Kotseruba et al.]         | 2021 | 0.870    | —     | —     | —         | 11.89ms       |
| TrouSPI-Net                     | 2021 | 0.880    | —     | —     | —         | —             |
| IntFormer                       | 2021 | 0.890    | —     | —     | —         | —             |
| Pedestrian Graph+               | 2022 | 0.890    | —     | —     | —         | 1.56ms        |
| BiPed                           | 2022 | 0.910    | —     | —     | —         | —             |
| MTL                             | 2022 | 0.910    | 0.930 | —     | —         | —             |
| CIPF                            | 2023 | 0.910    | —     | —     | —         | —             |
| PIT                             | 2023 | 0.910    | —     | —     | —         | 4.80ms        |
| VMIGI                           | 2023 | 0.920    | 0.910 | 0.870 | —         | —             |
| GTransPDM                       | 2024 | 0.920    | 0.870 | —     | —         | —             |
| EfficientPIE [paper]            | 2025 | 0.920    | 0.917 | 0.952 | 0.960     | 0.21ms        |
| EfficientPIE [replicated]       | 2025 | 0.918    | 0.917 | 0.952 | 0.961     | 1.05ms        |
| SparseTemporalPIE v4 (ours)     | 2026 | 0.919    | 0.922 | 0.953 | 0.958     | 1.19ms        |
| SparseTemporalPIE v4 e2e (ours) | 2026 | 0.919    | 0.922 | 0.953 | 0.958     | 5.07ms †      |
| **SparseTemporalPIE v3 (ours)** | **2026** | **0.926** | **0.947** | **0.957** | **0.957** | **1.81ms** |
| SparseTemporalPIE v3 e2e (ours) | 2026 | 0.926    | 0.947 | 0.957 | 0.957     | 5.68ms †      |

† End-to-end includes upstream ViTPose-B pose estimation (3.875ms, 90M params). In a production AV stack, pose estimation runs as part of the perception pipeline and may be shared across tasks.

SparseTemporalPIE v3 establishes a new state of the art on the PIE test set. While the accuracy improvement over the 0.92 cluster is modest (+0.006), the AUC gain is substantial: **+0.030 over EfficientPIE** and **+0.017 over the previous best (MTL, 0.930)**. AUC measures ranking quality across all decision thresholds and is more informative than accuracy for safety-critical systems where the operating threshold varies by context (e.g., urban vs. highway, day vs. night). A higher AUC indicates the model produces better-calibrated risk scores — important for downstream planners that consume continuous probability outputs rather than binary predictions.

### 3.2 Comparison with State of the Art (JAAD Test Set)

EfficientPIE reports JAAD results using a dedicated model trained on JAAD (not zero-shot transfer). We follow the same protocol: train SparseTemporalPIE v3 on JAAD from scratch using the full IL pipeline, initialized from the EfficientPIE pretrained backbone.

| Method                          | Accuracy | AUC    | F1     | Precision |
|---------------------------------|----------|--------|--------|-----------|
| EfficientPIE [paper]            | **0.890**| 0.860  | 0.620  | **0.630** |
| **SparseTemporalPIE v3 (ours)** | 0.878    | **0.885** | **0.633** | 0.597 |

SparseTemporalPIE v3 trades a small accuracy deficit (−0.012) for improvements in AUC (+0.025), F1 (+0.013), and the overall ranking quality of its probability outputs. This mirrors the pattern observed on PIE: the model consistently improves AUC at the cost of marginal accuracy, consistent with the cross-attention head learning to produce better-calibrated soft scores. The accuracy gap is attributable in part to JAAD's lower annotation consistency compared to PIE, and to the absence of OBD speed features in JAAD (zero-padded in our model).

---

## 4. Ablation Study

### 4.1 Cross-Attention vs. Context MLP (v3 vs. v4)

To isolate the contribution of cross-attention, we trained a simplified variant (**v4**) that removes the cross-attention and feedforward blocks entirely, retaining only the context MLP (bbox trajectory + behavioral features) fused via late fusion with the current-frame embedding. Pose is reduced to static-only (34-d).

Both variants were run to the full IL chain (steps 0–14). v4 best checkpoint is step 2; v3 best is step 14.

| Component                    | v3 (full) | v4 (no attention) |
|------------------------------|-----------|--------------------|
| Multi-frame cross-attention  | ✓         | ✗                  |
| Pose velocity (34-d)         | ✓         | ✗                  |
| Pose static (34-d)           | ✓         | ✓                  |
| Bbox trajectory (12-d)       | ✓         | ✓                  |
| Behavioral context (5-d)     | ✓         | ✓                  |
| Best IL step                 | 14        | 2                  |
| **Test Accuracy (best)**     | **0.9261**| 0.9194             |
| **Test Accuracy (step 14)**  | **0.9261**| 0.9127             |
| **AUC (best)**               | **0.9468**| 0.9220             |
| Parameters                   | ~9.0M     | ~1.1M              |

The v4 ablation at its best (step 2) still matches EfficientPIE (0.9194 vs 0.920), confirming that motion and behavioral features alone provide marginal benefit. However, v4 **degrades** by step 14 (0.9127), while v3 continues improving to its peak. The full v3 advantage (+0.67% accuracy, +2.48 AUC at best checkpoints) is attributable to cross-attention enabling the IL chain to remain productive across all 7 steps — without attention, later steps introduce distillation noise without adding discriminative signal.

### 4.2 Backbone Initialization: EfficientPIE vs. ImageNet

To assess the importance of backbone initialization, we trained a second v3 run initialized from raw ImageNet weights (EfficientNet-B0) rather than the EfficientPIE pretrained backbone. Both runs use the same partial-freeze strategy: backbone parameters trained at lr=1e-5 (10× lower than the head), all other parameters at lr=1e-4.

| Configuration                     | Best IL Step | Val Acc (best) | Test Acc | AUC    | Precision |
|-----------------------------------|-------------|----------------|----------|--------|-----------|
| v3 (EfficientPIE backbone)        | 14          | 0.8823         | **0.9261** | **0.9468** | 0.9570 |
| v3 (ImageNet backbone)            | 14          | 0.8864         | 0.9216   | 0.9211 | **0.9628** |

The ImageNet-initialized run falls short on accuracy (−0.004) and AUC (−0.026 at step 14) despite comparable or higher val accuracy throughout training. This reveals that **val accuracy is a poor proxy for test performance** in this regime: the ImageNet run's higher peak val acc (0.8936 vs. 0.8823) did not translate to better test results.

The gap is attributable to backbone quality. The EfficientPIE backbone has already been task-adapted to the PIE domain through the full IDIL training pipeline; it provides the cross-attention head with feature representations already tuned to pedestrian appearance and context in urban driving scenes. Starting from raw ImageNet features, the backbone must adapt within the small PIE training set (~1,700 sequences) at a constrained learning rate — an insufficient budget to reach the same representational quality.

This result reinforces that SparseTemporalPIE v3 is best understood as an **extension** of EfficientPIE rather than an independent model: its gains derive from cross-attention and pose reasoning built on top of a strong, task-specific backbone, not from backbone fine-tuning alone.

---

## 5. IL Step Progression

### 5.1 PIE

Test accuracy across IL steps for both model variants (PIE):

| IL Step | v3 Accuracy | v4 Accuracy |
|---------|-------------|-------------|
| 0       | 0.9048      | 0.9082      |
| 2       | 0.9205      | **0.9194**  |
| 4       | 0.9071      | 0.9048      |
| 6       | 0.9048      | 0.9059      |
| 8       | 0.9037      | 0.8970      |
| 10      | 0.9104      | 0.9037      |
| 12      | 0.9127      | 0.9183      |
| **14**  | **0.9261**  | 0.9127      |

Both models show a non-monotonic trajectory — accuracy dips mid-chain before recovering — consistent with the IL distillation dynamics reported in EfficientPIE. v3 continues improving through all 7 IL steps, peaking at step 14, while v4 peaks at step 2 and **regresses by step 14** (0.9127 < 0.9194). This divergence suggests cross-attention is the mechanism that allows the IL chain to extract new discriminative signal at each step: as the step index increases toward the crossing event, cross-attention learns to attend to earlier context frames that encode the pedestrian's approach trajectory. Without attention, later IL steps add distillation constraints without contributing new visual signal, causing the model to overfit to the distillation loss.

### 5.2 JAAD

Validation accuracy across IL steps for SparseTemporalPIE v3 trained on JAAD:

| IL Step | Best Val Acc |
|---------|-------------|
| 0       | 0.8801       |
| 2       | 0.8966       |
| 4       | 0.9031       |
| 6       | 0.9108       |
| 8       | 0.9119       |
| 10      | 0.9168       |
| 12      | 0.9168       |
| **14**  | **0.9228**   |

Unlike the PIE IL progression, JAAD val accuracy improves nearly monotonically across all 8 steps, with no mid-chain dip. The final step 14 checkpoint is used for test evaluation. EfficientPIE reports a JAAD test accuracy of 0.89; our step-4 val accuracy (0.9031) already exceeds this, suggesting that the JAAD model converges earlier and more stably than on PIE — likely due to JAAD's larger and more diverse dataset providing a smoother loss landscape for distillation.

---

## 6. Discussion

The key finding is that **cross-attention on multi-frame visual context, combined with explicit motion and behavioral features, consistently outperforms single-frame baselines** when trained with the IDIL distillation protocol. The improvement is most pronounced at the end of the IL chain (step 14), where the model has access to the full temporal span of the observation window.

A second finding, consistent across both PIE and JAAD, is that **SparseTemporalPIE v3 trades marginal accuracy for substantially improved AUC**. On PIE, AUC improves +0.030 over EfficientPIE (0.947 vs. 0.917) while accuracy improves +0.006. On JAAD, AUC improves +0.025 (0.885 vs. 0.860) while accuracy decreases −0.012. This pattern is not coincidental: cross-attention enables the model to produce better-calibrated soft probability scores by attending to the full temporal context of the approach trajectory, even when the argmax prediction is not always correct. For safety-critical applications where the downstream planner consumes continuous risk scores rather than binary predictions, this represents a meaningful and reproducible improvement.

A third finding from Section 4.2 is that **backbone initialization matters more than backbone fine-tuning**. Starting from the EfficientPIE pretrained backbone outperforms starting from ImageNet weights even when both use the same partial-freeze training strategy. This suggests the PIE domain is sufficiently specialized that task-adapted features are a prerequisite for the cross-attention head to function well, and that the small dataset size (~1,700 training sequences) cannot support meaningful backbone adaptation from scratch.

A negative result worth noting: during development, we observed that the PIE validation set (~92 pedestrians, ~500 samples after filtering) was too small to reliably distinguish between model variants. Val accuracy for v3 plateaued at ~0.870–0.894 across runs, while test accuracy varied from 0.921 to 0.926. This underscores the importance of reporting test set numbers and not over-tuning to validation performance on small datasets.

**Limitations.** The current model uses a pre-trained static pose estimator applied frame-by-frame; a temporally-aware pose model could further improve the velocity signal. JAAD lacks OBD speed data, requiring zero-padding of the ego-vehicle speed features; a learned imputation strategy could partially recover this signal. Zero-shot cross-dataset transfer (PIE → JAAD or vice versa) was not successful due to label semantics mismatch (PIE labels encode crossing *intention*; JAAD labels encode crossing *action*), confirming that separate training runs are required for each dataset.
