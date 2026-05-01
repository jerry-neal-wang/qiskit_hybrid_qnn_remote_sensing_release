# Regime-Bounded Hybrid Quantum Heads for Low-Shot Remote-Sensing ROI Classification

## Abstract

Hybrid quantum neural networks are increasingly explored for Earth-observation image analysis, but their empirical value remains difficult to interpret when quantum modules are not compared with matched classical controls or when positive results are not stress-tested outside a single experimental window. This paper presents a controlled Qiskit-based evaluation of hybrid quantum heads for low-shot remote-sensing region-of-interest (ROI) classification on DIOR-derived object crops. The main experiment uses the four-class `transport_logistics4` subset under a 16-shot protocol with seven paired random seeds. A lightweight CNN baseline, a matched classical hybrid control, a non-entangled quantum head, and a ZZFeatureMap plus RealAmplitudes quantum head are compared using macro-F1, paired two-sided Wilcoxon tests, paired bootstrap confidence intervals, per-class analysis, and simulation-side runtime tradeoffs. In the main low-shot window, the matched control achieves 0.3055 ± 0.1729 macro-F1, while the non-entangled and ZZ+RealAmplitudes heads reach 0.5031 ± 0.1421 and 0.4545 ± 0.1735, respectively. The two QNN-control deltas are positive across most paired seeds and have strictly positive bootstrap confidence intervals. However, the margin contracts at 32 shots, shows no significant QNN-control difference at 64 shots, is compressed under a ResNet18 transfer backbone, and is not stably reproduced on an additional `urban_structural4` subset. The resulting claim is deliberately bounded: Qiskit hybrid quantum heads can provide measurable benefit in a narrow low-shot DIOR ROI regime, but the effect depends on label count, backbone strength, subset composition, optimization policy, and runtime cost. The study is positioned as a boundary-aware quantum machine learning evaluation rather than a state-of-the-art remote-sensing classifier.

**Keywords:** hybrid quantum neural networks; Qiskit; remote sensing; DIOR; low-shot learning; ROI classification; matched control; quantum machine learning; variational quantum circuits

## 1. Introduction

Deep neural networks have become the dominant family of models for remote-sensing image analysis, including scene classification, object detection, semantic segmentation, and region-level recognition [1–10]. Their success, however, often depends on abundant labeled data and strong feature extractors. In optical remote sensing, labeled object-level examples can be costly to curate, class distributions are often uneven, and visual appearance varies with spatial resolution, object scale, viewpoint, background clutter, and acquisition conditions. These factors make low-shot remote-sensing ROI classification a useful setting for evaluating compact or unconventional classification heads.

Quantum machine learning (QML) has been proposed as a possible route for building compact nonlinear models through parameterized quantum circuits [19–28]. In hybrid quantum-classical learning, a classical neural network first extracts features, which are then projected into a parameterized quantum circuit whose expectation values are integrated with a classical classifier. This idea is attractive for low-dimensional bottleneck settings, where the quantum circuit can act as a structured nonlinear head. At the same time, current QML experiments are vulnerable to overinterpretation: a positive result may reflect additional head capacity, an unstable training protocol, or a favorable task subset rather than a general quantum benefit.

This paper studies the problem in a controlled remote-sensing setting. We evaluate Qiskit hybrid quantum heads on DIOR-derived ROI crops rather than on broad land-cover scenes. The main subset, `transport_logistics4`, contains airplane, ship, vehicle, and harbor targets. The main training protocol uses 16 labeled training crops per class, seven paired seeds, and a lightweight CNN backbone. The primary comparison is not simply quantum versus a plain CNN. Instead, each quantum head is compared against a matched classical hybrid control with the same general backbone, projection, fusion, and training structure. The purpose is to isolate whether replacing the matched branch with a quantum head changes seed-level macro-F1 under the same low-shot setting.

The most important prior benchmark for this direction is the circuit-based hybrid quantum neural network study on remote-sensing imagery by Sebastianelli et al. [29]. That work evaluated hybrid quantum CNNs on EuroSAT-style land-use and land-cover scene classification and emphasized circuit comparisons, including the role of entanglement. Our study is deliberately different. It moves from EuroSAT scene classification to DIOR object-level ROI classification, from a broad scene benchmark to a low-shot object-subset protocol, and from a quantum-versus-classical baseline comparison to a matched-control, multi-seed, boundary-aware evaluation. We also include larger-shot, strong-backbone, cross-subset, bootstrap, class-level, and runtime stress tests. The novelty is therefore not that a quantum layer is inserted into a remote-sensing classifier, but that the claimed gain is evaluated with explicit empirical boundaries.

The central question is: under what conditions, if any, do Qiskit hybrid quantum heads provide measurable benefit over a matched classical head for low-shot remote-sensing ROI classification?

The answer is bounded. In the main 16-shot `transport_logistics4` window, both quantum heads outperform the matched classical control in macro-F1 with paired statistical support. Outside that window, the margin contracts. At 32 and 64 shots, the difference from the control is not significant. Under a ResNet18 transfer backbone, the quantum-control gap is compressed. On a second subset, `urban_structural4`, the quantum heads do not stably exceed the control. We therefore do not claim universal quantum advantage. We claim a narrow, reproducible, and stress-tested low-shot window in which hybrid quantum heads can help.

The contributions are fourfold:

- We formulate DIOR-derived low-shot ROI classification as a controlled Qiskit hybrid-head evaluation problem rather than as a generic remote-sensing benchmark competition.
- We introduce a matched classical hybrid control so that quantum heads are not compared only against a weak CNN baseline.
- We report seven-seed macro-F1 results, paired Wilcoxon tests, paired bootstrap confidence intervals, class-level analysis, and runtime tradeoffs for two quantum heads.
- We identify the empirical boundaries of the observed gain across label count, backbone strength, and subset composition, showing that the benefit is regime-limited rather than universal.

## 2. Related Work

### 2.1 Remote-sensing benchmarks and ROI-level recognition

Remote-sensing image analysis has long relied on benchmark datasets that emphasize scene-level or object-level understanding. UC Merced, AID, EuroSAT, NWPU-RESISC45, DOTA, and DIOR have all contributed to this progression [1–7]. Scene datasets such as EuroSAT and NWPU-RESISC45 are well suited for land-cover or scene-label recognition, while object-level datasets such as DIOR and DOTA support target-centric evaluation. DIOR is particularly relevant here because it contains a large number of optical remote-sensing object instances across 20 categories, with substantial object-scale variation, intra-class diversity, and inter-class similarity [1].

This paper uses DIOR-derived ROI crops rather than full-scene classification images. That distinction is important. Scene classification can often exploit global contextual cues, whereas object-level ROI classification requires the classifier to separate local target appearance under scale variation and background clutter. The `transport_logistics4` subset used in this study is therefore closer to fine-grained object-level recognition than to EuroSAT-style land-use scene classification.

### 2.2 Low-shot and few-shot remote-sensing learning

Limited-label learning is a persistent issue in remote sensing because manual annotation is expensive and class distributions can be long-tailed. Few-shot remote-sensing scene classification has been studied through transfer learning, metric learning, meta-learning, self-supervision, prompt learning, and feature-fusion strategies [11–18]. Recent surveys emphasize that remote-sensing few-shot learning differs from natural-image few-shot learning because of high intra-class variance, small inter-class differences, domain shift, and scale ambiguity [11,14].

Our study does not propose a new meta-learning algorithm. Instead, it uses low-shot protocols as a controlled regime in which compact hybrid heads can be compared. The focus is on whether quantum heads provide additional macro-F1 over a matched classical head when the backbone and training protocol are held fixed. This makes the paper a controlled QML evaluation rather than a general few-shot remote-sensing method.

### 2.3 Hybrid quantum neural networks and variational circuits

Quantum machine learning has developed around the idea that quantum feature maps, variational circuits, and quantum kernels may provide useful transformations for selected learning problems [19,24–28]. Hybrid variational algorithms combine parameterized quantum circuits with classical optimization and are attractive for noisy intermediate-scale quantum settings [20]. However, trainability and scalability remain open challenges. Barren plateaus, noise-induced gradient suppression, circuit expressibility, and runtime overhead can limit practical gains [21–23].

Hybrid quantum neural networks are often evaluated by inserting a quantum layer into a classical model. Such comparisons can be misleading if the classical baseline lacks a matched architectural control. The present study therefore includes both a plain CNN baseline and a matched hybrid classical control. This distinction follows the broader methodological principle that QML claims should be tested against controls that isolate architecture capacity from specifically quantum components.

### 2.4 Quantum machine learning for Earth observation

The direct benchmark paper for this study is Sebastianelli et al. [29], which used circuit-based hybrid quantum neural networks for remote-sensing imagery classification on EuroSAT and compared alternative circuit designs. Zhang et al. [30] later investigated hybrid classical-quantum transfer learning CNNs with small samples for remote-sensing scene classification. Fan et al. [31] proposed a hybrid quantum deep learning model with superpixel encoding for Earth-observation data classification, showing another route for reducing quantum encoding overhead.

The present paper differs in three ways. First, the task is DIOR-derived object-level ROI classification rather than EuroSAT-style scene classification. Second, the paper emphasizes matched controls and paired seed-level statistics rather than only model-level performance. Third, the study explicitly maps where the quantum-control margin collapses: larger label counts, stronger backbones, different subsets, and higher runtime. This makes the contribution a regime-boundary evaluation of hybrid QML in remote sensing.

### 2.5 Qiskit-based hybrid learning

The experiments use Qiskit Machine Learning through `EstimatorQNN` and `TorchConnector` [32–34]. The EstimatorQNN represents a parameterized quantum circuit with designated input and weight parameters, optional observables, and expectation-value outputs [33]. TorchConnector exposes a Qiskit neural network as a PyTorch module and requires input gradients to be enabled for backpropagation through the hybrid model [34]. PyTorch and scikit-learn are used for the surrounding neural training and metrics [35,36].

## 3. Methodology

### 3.1 Dataset construction and subset definition

We construct ROI crops from DIOR object annotations and preserve source-image train/validation/test partitions. Low-shot sampling is applied only to the training split. Validation and test evaluation follow the final full-split protocol, while a capped validation set is used during training to reduce simulation-side cost. The main subset, `transport_logistics4`, contains airplane, ship, vehicle, and harbor. A second subset, `urban_structural4`, contains vehicle, bridge, harbor, and storagetank and is used as a cross-subset boundary test.

Subset selection was exploratory before the final seven-seed confirmation. A candidate-screening stage evaluated semantically coherent subsets and identified `transport_logistics4` as the main positive low-shot window. The final claim is therefore not framed as a pre-registered discovery across all DIOR subsets. Instead, the seven-seed main result is treated as a confirmation of a selected regime, and the second subset is used to prevent overgeneralization. Supplementary Table S2 records the subset-screening context and labels which runs are main evidence, boundary evidence, or exploratory/superseded.

### 3.2 Evaluation ladder

The study is organized as an evaluation ladder rather than a single isolated win (Fig. 1). The first level tests the main low-shot window: `transport_logistics4`, small CNN backbone, 16 shots, and seven seeds. The second level increases label count to 32 and 64 shots. The third level replaces the lightweight backbone with ImageNet-pretrained ResNet18. The fourth level tests the same 16-shot small-CNN protocol on `urban_structural4`. This design allows the paper to state not only where a quantum-control margin appears but also where it vanishes.

![Fig. 1. Regime-bounded study design.](../artifacts/paper_composite_figures/fig1_study_design_overview.png)

**Fig. 1. Regime-bounded study design.** The evaluation ladder starts from a selected main low-shot window and then stress-tests the same claim across label count, backbone strength, and subset composition. The matched-control comparison is placed before boundary tests so that the quantum head is not evaluated only against a weak CNN baseline.

### 3.3 Model variants

Four model variants are compared. The CNN baseline is a lightweight classifier that directly maps the small-CNN backbone feature to four logits. The matched classical hybrid control uses the same backbone and projector as the quantum variants but replaces the quantum circuit with a classical branch. The non-entangled QNN head uses a four-qubit circuit with Ry data reuploading and four Z expectation outputs. The ZZ+RealAmplitudes head uses a ZZFeatureMap with one repetition and a RealAmplitudes ansatz with one repetition. Both QNN heads scale their four projected inputs as `tanh(x) * pi/2` before quantum encoding.

The hybrid fusion mechanism concatenates the 64-dimensional small-CNN feature with the 4-dimensional QNN or control output, applies a linear fusion classifier, and adds residual classical logits. The matched classical control is therefore dimension- and protocol-matched, but it is not claimed to be exactly parameter-matched to the quantum circuits. Table 1 summarizes the reproducibility-grade protocol, and Table 2 summarizes the architecture interface and trainable parameter counts.

**Table 1. Reproducibility-grade summary of the experimental protocol.**

| Item | Specification |
| --- | --- |
| Dataset source | DIOR-derived ROI crops from object annotations; source-image split preserved |
| Main subset | transport_logistics4: airplane, ship, vehicle, harbor |
| Boundary subset | urban_structural4: vehicle, bridge, harbor, storagetank |
| Input image size | 64 × 64 RGB crops |
| Shots and seeds | 16/32/64 shots per class; seeds 42–48; main confirmation uses 7 paired seeds |
| Low-shot sampling | Performed only on the training split; validation/test evaluation uses the final full split protocol |
| Training-time validation cap | 256 samples per class during training for efficient model selection |
| Backbones | Small CNN for the mainline; ImageNet-pretrained ResNet18 for the strong-backbone boundary |
| Methods | CNN baseline; matched classical hybrid control; QNN No-Ent; QNN ZZ+RealAmp |
| Optimization | Adam, learning rate 1e-3, weight decay 1e-4, cosine schedule with one warmup epoch |
| Epochs and batches | Baseline 10 epochs, hybrid/control 12 epochs; batch size 32 for baseline and 16 for hybrid/control |
| Hybrid stabilization | Backbone frozen for first 2 hybrid epochs; backbone learning-rate scale 0.1 during joint fine-tuning |
| Regularization | Dropout 0.2 in projection layers; gradient clipping at 1.0; early-stopping patience 4 |
| Quantum settings | 4 qubits, 4 QNN outputs, feature-map reps = 1, ansatz reps = 1, multi-Z observables |
| Primary metric/statistics | Macro-F1; paired two-sided Wilcoxon signed-rank tests; paired bootstrap confidence intervals |

**Table 2. Architecture and parameter-count summary.**

All hybrid heads first project the backbone feature vector with `Linear(D,32) -> GELU -> Dropout(0.2) -> Linear(32,4)`. QNN inputs are scaled as `tanh(x) * pi/2`. The fusion formula below uses `D` for the backbone feature dimension and a 4-dimensional branch output.

| Model | Backbone | D | Branch | Fusion/classifier | Trainable params |
| --- | --- | --- | --- | --- | --- |
| CNN Baseline | small cnn | 64 | None | Linear(64,4) | 61,060 |
| Hybrid Classical Control | small cnn | 64 | Linear(4,4)+Tanh | Concat(D,4)->Linear(68,4)+residual Linear(64,4) | 63,588 |
| QNN No-Ent | small cnn | 64 | 4-qubit Ry, 4 Z outputs | Concat(D,4)->Linear(68,4)+residual Linear(64,4) | 63,572 |
| QNN ZZ+RealAmp | small cnn | 64 | ZZFeatureMap+RealAmp, 4 Z outputs | Concat(D,4)->Linear(68,4)+residual Linear(64,4) | 63,576 |
| CNN Baseline | resnet18 pretrained | 128 | None | Linear(128,4) | 11,242,692 |
| Hybrid Classical Control | resnet18 pretrained | 128 | Linear(4,4)+Tanh | Concat(D,4)->Linear(132,4)+residual Linear(128,4) | 11,247,524 |
| QNN No-Ent | resnet18 pretrained | 128 | 4-qubit Ry, 4 Z outputs | Concat(D,4)->Linear(132,4)+residual Linear(128,4) | 11,247,508 |
| QNN ZZ+RealAmp | resnet18 pretrained | 128 | ZZFeatureMap+RealAmp, 4 Z outputs | Concat(D,4)->Linear(132,4)+residual Linear(128,4) | 11,247,512 |

### 3.4 Training protocol

For the main small-CNN experiments, the CNN baseline is trained first. Hybrid variants then warm-start from the corresponding baseline checkpoint. During hybrid training, the backbone is frozen for the first two epochs and jointly fine-tuned afterward with a backbone learning-rate scale of 0.1. This differential backbone learning rate is treated as part of the method rather than as a minor implementation detail, because earlier exploratory runs showed that uniform joint fine-tuning destabilized the hybrid heads.

The main experiments use image size 64, training augmentation `train_light`, Adam optimization with learning rate 1e-3 and weight decay 1e-4, cosine scheduling with one warmup epoch, gradient clipping at 1.0, and early stopping with patience 4. Baseline batch size is 32, while hybrid/control batch size is 16. The primary metric is macro-F1 on final test evaluation.

### 3.5 Statistical analysis

All main comparisons are paired by seed. We use two-sided Wilcoxon signed-rank tests for QNN-control comparisons and paired bootstrap confidence intervals over seed-level macro-F1 deltas. The paired design is used because each method is evaluated on the same seeds and low-shot sampling protocol. Seven paired seeds make p-values coarse; therefore, p-values are interpreted together with mean deltas, median deltas, better/worse seed counts, and bootstrap intervals. Unless explicitly stated, p-values are reported without multiple-comparison correction because the boundary tests are exploratory and diagnostic. Under a simple two-comparison Bonferroni correction for the two main QNN-control tests, the ZZ+RealAmplitudes comparison remains below the corrected 0.025 threshold, while the non-entangled comparison is borderline by Wilcoxon but remains supported by a strictly positive bootstrap interval.

## 4. Experiments and Results

### 4.1 Main 16-shot evidence on `transport_logistics4`

The main 16-shot evidence is shown in Fig. 2 and Table 3. Under the stabilized low-shot protocol, both QNN heads improve over the matched classical control in macro-F1. The classical control reaches 0.3055 ± 0.1729, whereas the non-entangled head reaches 0.5031 ± 0.1421 and the ZZ+RealAmplitudes head reaches 0.4545 ± 0.1735. The plain CNN baseline remains weaker at 0.2650 ± 0.0863.

The paired seed-level results clarify the robustness of this main window. The non-entangled QNN improves over the control in 6 of 7 seeds, with a mean delta of +0.1976 and a median delta of +0.1256. The ZZ+RealAmplitudes head improves over the control in all 7 seeds, with a mean delta of +0.1491 and a median delta of +0.1212. Paired bootstrap intervals are strictly positive for both comparisons.

**Table 3. Main 16-shot results on `transport_logistics4` across seven seeds.**

| Method | Macro-F1 | Accuracy | Runtime (s) | Interpretation |
| --- | --- | --- | --- | --- |
| CNN baseline | 0.2650 ± 0.0863 | 0.4715 ± 0.1638 | 52.66 ± 3.93 | Plain lightweight CNN reference |
| Hybrid classical control | 0.3055 ± 0.1729 | 0.4134 ± 0.1301 | 59.54 ± 5.49 | Matched classical hybrid head |
| QNN No-Ent | 0.5031 ± 0.1421 | 0.5451 ± 0.1281 | 137.20 ± 8.30 | Positive main-window quantum head |
| QNN ZZ+RealAmp | 0.4545 ± 0.1735 | 0.5108 ± 0.1339 | 328.08 ± 26.84 | Positive but slower entangled head |

**Table 4. Paired QNN-control statistics for the main 16-shot window.**

| Comparison | Pairs | Mean delta | Median delta | Better/Worse/Tie | Wilcoxon p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| QNN No-Ent - control | 7 | +0.1976 | +0.1256 | 6/1/0 | 0.03125 | [+0.0766, +0.3271] |
| QNN ZZ+RealAmp - control | 7 | +0.1491 | +0.1212 | 7/0/0 | 0.01562 | [+0.0621, +0.2665] |

![Fig. 2. Main 16-shot evidence on transport_logistics4.](../artifacts/paper_composite_figures/fig2_main_lowshot_evidence.png)

**Fig. 2. Main 16-shot evidence on `transport_logistics4`.** Seed-level macro-F1, paired QNN-control deltas, optimization dynamics, class-level F1, and mean confusion matrices. The optimization curves are interpreted as consistency evidence rather than causal proof.

### 4.2 Class-structured effect

The per-class pattern shows that the gain is not uniform across all four classes. The strongest improvements over the control occur in airplane, ship, and harbor, while vehicle contributes less to the quantum-control margin. This matters because vehicle is comparatively easier in this subset, whereas airplane, ship, and harbor account for more of the low-shot confusion. The class-structured pattern supports a boundary interpretation: quantum heads appear useful for selected difficult class boundaries rather than for uniformly lifting all categories.

### 4.3 Label-count boundary

The same subset and small-CNN protocol were evaluated at 32 and 64 shots. At 32 shots, the quantum-control deltas shrink to +0.0094 for the non-entangled head and +0.0018 for ZZ+RealAmplitudes, with both Wilcoxon p-values equal to 0.8125. At 64 shots, the non-entangled delta is -0.0021 and the ZZ+RealAmplitudes delta is +0.0001, again with no significant QNN-control difference. Thus, the positive margin does not scale cleanly with label count. It is a low-shot window, not a broad superiority effect.

### 4.4 Bootstrap boundary validation

The paired bootstrap analysis agrees with the Wilcoxon results. At 16 shots, the non-entangled head has a 95% bootstrap confidence interval of [+0.0766, +0.3271] for its macro-F1 delta over the control, and the ZZ+RealAmplitudes head has a 95% interval of [+0.0621, +0.2665]. At 64 shots, the intervals cross zero: [-0.0144, +0.0068] for non-entangled and [-0.0045, +0.0058] for ZZ+RealAmplitudes. This confirms that the 16-shot window is positive, while the 64-shot margin is effectively centered near zero.

### 4.5 Strong-backbone and cross-subset boundaries

The ResNet18 boundary tests whether the quantum-control margin survives a stronger feature extractor. It does not. Under `transport_logistics4 + ResNet18 + shot=16`, the matched control reaches 0.8545 ± 0.0059, the non-entangled head reaches 0.8534 ± 0.0027, and ZZ+RealAmplitudes reaches 0.8596 ± 0.0125. Neither quantum head significantly exceeds the control. The mean differences are small and seed-sign patterns are mixed.

The cross-subset boundary shows the same caution. On `urban_structural4`, the control reaches 0.4655 ± 0.1655, the non-entangled head reaches 0.4593 ± 0.1951, and the ZZ+RealAmplitudes head reaches 0.4073 ± 0.1658. Both QNN-control comparisons are non-significant. This confirms that the positive `transport_logistics4` window should not be generalized across DIOR-derived subsets.

**Table 5. Boundary experiments for label count, backbone strength, and subset composition.**

| Boundary | Setting | Control | QNN No-Ent | QNN ZZ+RealAmp | No-Ent delta / p | ZZ delta / p |
| --- | --- | --- | --- | --- | --- | --- |
| Shot boundary | transport_logistics4 / small CNN / shot=32 | 0.6034 ± 0.0478 | 0.6128 ± 0.0557 | 0.6053 ± 0.0239 | +0.0094; p=0.8125 | +0.0018; p=0.8125 |
| Shot boundary | transport_logistics4 / small CNN / shot=64 | 0.6705 ± 0.0378 | 0.6685 ± 0.0280 | 0.6707 ± 0.0314 | -0.0021; p=0.8125 | +0.0001; p=0.9375 |
| Backbone boundary | transport_logistics4 / ResNet18 / shot=16 | 0.8545 ± 0.0059 | 0.8534 ± 0.0027 | 0.8596 ± 0.0125 | -0.0011; p=0.6875 | +0.0051; p=0.5781 |
| Subset boundary | urban_structural4 / small CNN / shot=16 | 0.4655 ± 0.1655 | 0.4593 ± 0.1951 | 0.4073 ± 0.1658 | -0.0063; p=0.6875 | -0.0583; p=0.1563 |

**Table 6. Seed-level delta summary across main and boundary settings.**

| Setting | QNN head | Mean delta | Median delta | Better/Worse/Tie | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| transport small-CNN shot16 bblr0.1 | No-Ent | +0.1976 | +0.1256 | 6/1/0 | 0.03125 |
| transport small-CNN shot16 bblr0.1 | ZZ+RealAmp | +0.1491 | +0.1212 | 7/0/0 | 0.01562 |
| transport small-CNN shot32 bblr0.1 | No-Ent | +0.0094 | -0.0031 | 3/4/0 | 0.81250 |
| transport small-CNN shot32 bblr0.1 | ZZ+RealAmp | +0.0018 | +0.0223 | 4/3/0 | 0.81250 |
| transport small-CNN shot64 bblr0.1 | No-Ent | -0.0021 | +0.0001 | 4/3/0 | 0.81250 |
| transport small-CNN shot64 bblr0.1 | ZZ+RealAmp | +0.0001 | -0.0002 | 3/4/0 | 0.93750 |
| transport ResNet18 shot16 bblr0.1 | No-Ent | -0.0011 | -0.0029 | 2/5/0 | 0.68750 |
| transport ResNet18 shot16 bblr0.1 | ZZ+RealAmp | +0.0051 | -0.0000 | 3/4/0 | 0.57812 |
| urban small-CNN shot16 bblr0.1 | No-Ent | -0.0063 | -0.0087 | 2/5/0 | 0.68750 |
| urban small-CNN shot16 bblr0.1 | ZZ+RealAmp | -0.0583 | -0.0604 | 2/5/0 | 0.15625 |

![Fig. 3. Boundary map of regime-limited quantum gains.](../artifacts/paper_composite_figures/fig3_boundary_regime_map.png)

**Fig. 3. Boundary map of regime-limited quantum gains.** Cross-condition QNN-control deltas, shot-wise contraction, bootstrap intervals, and runtime-vs-gain tradeoff. The figure shows that the quantum-control margin appears only in a narrow low-shot window and contracts under larger-shot, stronger-backbone, and cross-subset settings.

### 4.6 Runtime tradeoff

Runtime imposes an operational boundary. In the main window, the control requires 59.54 s on average, compared with 137.20 s for the non-entangled head and 328.08 s for ZZ+RealAmplitudes. Since ZZ+RealAmplitudes is much slower and does not consistently outperform the non-entangled head, the non-entangled variant currently provides the more favorable performance-runtime compromise. These timings are simulation-side costs and should not be interpreted as quantum hardware runtime.

## 5. Discussion

### 5.1 What the evidence supports

The evidence supports a bounded positive claim. In the selected `transport_logistics4` 16-shot window, Qiskit hybrid quantum heads improve over a matched classical control. This conclusion is supported by seven paired seeds, positive mean deltas, positive median deltas in the main comparisons, better/worse counts favoring the QNN heads, Wilcoxon tests, and bootstrap intervals. The result is stronger than a plain CNN comparison because the matched control shares the main hybrid architecture and training protocol.

### 5.2 What the evidence does not support

The evidence does not support universal quantum advantage. The QNN-control margin contracts with more labels, is compressed by a stronger ResNet18 backbone, and does not reproduce on `urban_structural4`. The results also do not support a simple entanglement narrative. In the main setting, the non-entangled head has the highest mean macro-F1 and the better runtime profile. The entangled ZZ+RealAmplitudes head is useful as a circuit comparison, but it is not consistently better than the non-entangled head and does not retain a significant margin over the matched control under boundary tests.

The results also do not imply practical quantum advantage. Training is simulation-side and CPU-based, and the quantum branches are slower than the control. The paper therefore avoids claims about deployment readiness or hardware advantage. Hardware-aware execution remains future work.

### 5.3 Why the benchmark-paper comparison matters

Compared with the EuroSAT hybrid-QCNN benchmark [29], the present study is more limited in scope but stricter in evaluation. The benchmark paper emphasized circuit-based hybrid classification for land-use/land-cover scenes and reported circuit-dependent gains, including stronger behavior from entangled circuits. Our study uses object-level DIOR ROI crops and explicitly asks whether quantum heads remain useful when matched classical controls and boundary tests are introduced. The conclusion is not that our method is superior to the benchmark paper, but that a different and more conservative evaluation question is needed for current QML claims in remote sensing.

### 5.4 Subset selection and selection-bias control

The subset-screening context is not treated as additional confirmatory evidence in the main text. The full chronology, including exploratory and superseded runs, is provided in Supplementary Table S2. In brief, `transport_logistics4` was selected after exploratory screening and then re-evaluated under the fixed seven-seed protocol, while `urban_structural4` is retained as a negative cross-subset boundary test.

### 5.5 Classical baselines and paper positioning

A remote-sensing reviewer may ask why the study does not compare against the strongest modern classifiers. The reason is that the paper is not intended as a state-of-the-art classification benchmark. ResNet18 is included as a strong-backbone boundary precisely to show that stronger classical features compress the quantum-control margin. The central baseline is the matched classical hybrid control, not a leaderboard model. Future work should add frozen-feature linear and MLP heads to further reduce classical-baseline vulnerability, but the current evidence is already sufficient for the narrower claim that a QNN-control margin appears only in a specific low-shot window.

## 6. Limitations

This study has several limitations. First, it uses DIOR-derived ROI subsets only. The observed boundary pattern may not hold on other Earth-observation datasets, sensors, or spatial resolutions. Second, subset selection remains a risk despite the inclusion of a cross-subset boundary test. Third, the experiments use simulator-side Qiskit training rather than hardware execution. Runtime reported here is therefore a simulation-side cost, not a quantum hardware benchmark. Fourth, the circuit family is limited to a non-entangled circuit and a ZZFeatureMap plus RealAmplitudes design. Fifth, the classical baseline set is controlled but not exhaustive; additional frozen-feature linear and MLP heads would further strengthen the submission. Sixth, because the analysis uses seven paired seeds, statistical conclusions should be interpreted as support for a controlled experimental pattern rather than as broad distributional claims.

## 7. Conclusion

This paper evaluated Qiskit hybrid quantum heads for low-shot DIOR-derived remote-sensing ROI classification under a matched-control and boundary-aware design. In the main `transport_logistics4` 16-shot setting, both a non-entangled quantum head and a ZZFeatureMap plus RealAmplitudes head outperform a matched classical hybrid control in macro-F1 across seven paired seeds. Paired Wilcoxon tests and paired bootstrap confidence intervals support this positive low-shot window.

The effect is bounded. It contracts at 32 shots, shows no significant QNN-control difference at 64 shots, is compressed under a ResNet18 transfer backbone, and is not stably reproduced on the `urban_structural4` subset. The runtime tradeoff also favors the non-entangled head over the more expensive ZZ+RealAmplitudes circuit in the main window. The final claim is therefore not universal quantum advantage. It is that hybrid quantum heads can help in a narrow low-shot remote-sensing ROI regime when evaluated against a matched classical control, and that this benefit is strongly conditioned by data regime, backbone strength, subset composition, optimization policy, and runtime cost.

Future work should extend the evaluation to additional datasets, define multiple subset families before training, include more classical-head baselines, test additional circuit families under the same matched-control protocol, and evaluate whether the observed regime-boundary pattern persists under hardware-aware quantum execution.

## Data and Code Availability

Code and processed metadata used to reproduce the reported tables and figures will be made available in an anonymized review package at [URL], release tag [TAG]. DIOR images are not redistributed; users should obtain DIOR from the original provider and run the preprocessing script described in `REPRODUCIBILITY.md`. The review package includes source code, environment files, clean result tables, seed-level CSVs, and figure-generation scripts.

## Supplementary Material

Supplementary Table S1: per-seed paired macro-F1 deltas for QNN No-Ent vs control and QNN ZZ+RealAmp vs control across all main and boundary settings.

Supplementary Table S2: subset and boundary-screening context, including which runs are main evidence and which are exploratory or superseded.

Supplementary Table S3: per-seed summary/detail tables from each final seven-seed artifact root.

Supplementary Table S4: paired Wilcoxon tables for all method pairs and boundary experiments.

Supplementary Table S5: paired bootstrap confidence intervals for QNN-control deltas.

Supplementary Table S6: per-class metrics for the main `transport_logistics4` 16-shot setting.

Supplementary figures may include additional confusion matrices, training curves, and early-stopping curves by seed. Main figures should remain Fig. 1–3.

## References

[1] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 159, pp. 296–307, 2020, doi: 10.1016/j.isprsjprs.2019.11.023.

[2] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017, doi: 10.1109/JPROC.2017.2675998.

[3] G.-S. Xia, J. Hu, F. Hu, B. Shi, X. Bai, Y. Zhong, L. Zhang, and X. Lu, “AID: A benchmark data set for performance evaluation of aerial scene classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 7, pp. 3965–3981, 2017, doi: 10.1109/TGRS.2017.2685945.

[4] Y. Yang and S. Newsam, “Bag-of-visual-words and spatial extensions for land-use classification,” in Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, 2010, pp. 270–279, doi: 10.1145/1869790.1869829.

[5] P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217–2226, 2019, doi: 10.1109/JSTARS.2019.2918242.

[6] G.-S. Xia et al., “DOTA: A large-scale dataset for object detection in aerial images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 3974–3983, doi: 10.1109/CVPR.2018.00418.

[7] G. Cheng and J. Han, “A survey on object detection in optical remote sensing images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 117, pp. 11–28, 2016, doi: 10.1016/j.isprsjprs.2016.03.014.

[8] X. X. Zhu et al., “Deep learning in remote sensing: A comprehensive review and list of resources,” IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 4, pp. 8–36, 2017, doi: 10.1109/MGRS.2017.2762307.

[9] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, pp. 436–444, 2015, doi: 10.1038/nature14539.

[10] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[11] C. Qiu, X. Zhang, X. Tong, N. Guan, X. Yi, K. Yang, J. Zhu, and A. Yu, “Few-shot remote sensing image scene classification: Recent advances, new baselines, and future trends,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 209, pp. 368–382, 2024, doi: 10.1016/j.isprsjprs.2024.02.005.

[12] Z. Yuan, C. Tang, A. Yang, W. Huang, and W. Chen, “Few-shot remote sensing image scene classification based on metric learning and local descriptors,” Remote Sensing, vol. 15, no. 3, 831, 2023, doi: 10.3390/rs15030831.

[13] T. Zhang, X. Zhang, P. Zhu, X. Jia, X. Tang, and L. Jiao, “Generalized few-shot object detection in remote sensing images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 195, pp. 353–364, 2023, doi: 10.1016/j.isprsjprs.2022.12.004.

[14] G. Y. Lee, T. Dam, M. M. Ferdaus, D. P. Poenar, and V. N. Duong, “Unlocking the capabilities of explainable few-shot learning in remote sensing,” Artificial Intelligence Review, vol. 57, article 169, 2024, doi: 10.1007/s10462-024-10803-5.

[15] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in Proceedings of the International Conference on Machine Learning, 2017, pp. 1126–1135.

[16] O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra, and K. Kavukcuoglu, “Matching networks for one shot learning,” in Advances in Neural Information Processing Systems, 2016.

[17] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for few-shot learning,” in Advances in Neural Information Processing Systems, 2017.

[18] F. Sung et al., “Learning to compare: Relation network for few-shot learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 1199–1208.

[19] J. Biamonte et al., “Quantum machine learning,” Nature, vol. 549, pp. 195–202, 2017, doi: 10.1038/nature23474.

[20] M. Cerezo et al., “Variational quantum algorithms,” Nature Reviews Physics, vol. 3, pp. 625–644, 2021, doi: 10.1038/s42254-021-00348-9.

[21] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, “Barren plateaus in quantum neural network training landscapes,” Nature Communications, vol. 9, 4812, 2018, doi: 10.1038/s41467-018-07090-4.

[22] S. Wang et al., “Noise-induced barren plateaus in variational quantum algorithms,” Nature Communications, vol. 12, 6961, 2021, doi: 10.1038/s41467-021-27045-6.

[23] M. Larocca et al., “Barren plateaus in variational quantum computing,” Nature Reviews Physics, vol. 7, pp. 174–189, 2025, doi: 10.1038/s42254-025-00813-9.

[24] V. Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” Nature, vol. 567, pp. 209–212, 2019, doi: 10.1038/s41586-019-0980-2.

[25] M. Schuld and N. Killoran, “Quantum machine learning in feature Hilbert spaces,” Physical Review Letters, vol. 122, 040504, 2019, doi: 10.1103/PhysRevLett.122.040504.

[26] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, “Quantum circuit learning,” Physical Review A, vol. 98, 032309, 2018, doi: 10.1103/PhysRevA.98.032309.

[27] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, “Parameterized quantum circuits as machine learning models,” Quantum Science and Technology, vol. 4, no. 4, 043001, 2019, doi: 10.1088/2058-9565/ab4eb5.

[28] E. Farhi and H. Neven, “Classification with quantum neural networks on near term processors,” arXiv:1802.06002, 2018.

[29] A. Sebastianelli, D. A. Zaidenberg, D. Spiller, B. Le Saux, and S. L. Ullo, “On circuit-based hybrid quantum neural networks for remote sensing imagery classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 565–580, 2022, doi: 10.1109/JSTARS.2021.3134785.

[30] Z. Zhang, X. Mi, J. Yang, X. Wei, Y. Liu, J. Yan, P. Liu, X. Gu, and T. Yu, “Remote sensing image scene classification in hybrid classical-quantum transferring CNN with small samples,” Sensors, vol. 23, no. 18, 8010, 2023, doi: 10.3390/s23188010.

[31] F. Fan, Y. Shi, T. Guggemos, X. X. Zhu, and X. Zhu, “Hybrid quantum deep learning with superpixel encoding for Earth observation data classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 6, pp. 11271–11284, 2025, doi: 10.1109/TNNLS.2024.3518108.

[32] M. E. Sahin et al., “Qiskit Machine Learning: An open-source library for quantum machine learning tasks at scale on quantum hardware and classical simulators,” arXiv:2505.17756, 2025.

[33] Qiskit Community, “EstimatorQNN,” Qiskit Machine Learning 0.9.0 Documentation, 2026. Available: https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html. Accessed: May 1, 2026.

[34] Qiskit Community, “TorchConnector,” Qiskit Machine Learning 0.9.0 Documentation, 2026. Available: https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.connectors.TorchConnector.html. Accessed: May 1, 2026.

[35] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems, 2019.

[36] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[37] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, vol. 1, no. 6, pp. 80–83, 1945, doi: 10.2307/3001968.

[38] B. Efron, “Bootstrap methods: Another look at the jackknife,” The Annals of Statistics, vol. 7, no. 1, pp. 1–26, 1979, doi: 10.1214/aos/1176344552.
