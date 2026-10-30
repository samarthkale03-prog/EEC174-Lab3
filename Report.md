# EEC 174AY Lab 3 Report — CNNs & Image Classification

**Students:** Samarth Kale, Saarth Kishore  
**Course:** EEC 174AY, UC Davis  

This report summarizes PreLab, Phase 1 (MNIST CNNs), Phase 2 (CIFAR-10 transfer learning), and Phase 3 (Stanford Cars make classification). All numbers come from the completed notebook runs; no training code was changed for this write-up.

---

## Results at a glance

| Phase | Task | Best model | Test accuracy |
|---|---|---|---|
| 1 | MNIST digit classification | Net5 + Batch Norm | **99.10%** |
| 2 | CIFAR-10 transfer learning | ResNet18, fully fine-tuned + augmentation | **85.87%** |
| 3 | Stanford Cars, by make (49 classes) | ConvNeXt-Tiny, fully fine-tuned | **95.83%** |

---

## PreLab

Implemented a one-hidden-layer network (3 → 6 ReLU → 4 sigmoid) with manual ReLU/sigmoid and a NumPy forward pass. Also practiced image I/O on `messi.jpeg`: RGB split, grayscale, normalization, 500×500 center crop, and a hand-written Sobel convolution. These basics support the CNN and vision work in later phases.

---

## Phase 1 — CNNs on MNIST

Built Net / Net2 / Net5 / FCN, trained for **10 epochs**, and compared hyperparameters. Default setup for most runs: Adam, best learning rate from Section 5.2, ReLU, batch size 32 unless noted.

### Q 5.1.1 — Learning curves
A learning curve plots loss and accuracy over training for train and test sets. It shows whether the model is still improving, has converged, or is under/overfitting, and whether the learning rate is reasonable. In Section 4, the 1-layer model’s loss dropped quickly then flattened (~94% accuracy); rising test loss with flat test accuracy was an early sign of mild overfitting at lr = 0.01.

### Q 5.2.1 — Learning rates (Net2)

| Learning rate | Test accuracy | Assessment |
|---|---|---|
| 5e-2 | **88.94%** | Too high — noisy loss, underfits (overshoots) |
| 2e-3 | **97.78%** | Trains well; slight late overfitting |
| 3e-4 | **97.83%** | Best — smooth curves, highest accuracy |

High LRs take large steps and can oscillate; low LRs are stable but slow. **Best LR: 3e-4.**

### Q 5.3.1 — Optimizers (lr = 3e-4)

| Optimizer | Test accuracy |
|---|---|
| SGD | **93.42%** |
| SGD + momentum 0.5 | **94.78%** |
| SGD + momentum 0.9 | **97.44%** |
| Adam | **97.83%** |

SGD uses a fixed step size; momentum smooths updates; Adam adds per-parameter adaptive rates. Adam was best; SGD was held back by an LR tuned for Adam. **Best optimizer: Adam.**

### Q 5.4.1 — Weight initialization

| Init | Test accuracy |
|---|---|
| All zeros | **11.35%** (no learning; symmetry) |
| All ones | **92.55%** (unstable; identical filters) |
| Normal(0, 0.01) | **98.09%** |

**Recommendation:** small random / Kaiming (He) init (PyTorch default for ReLU). Zeros break learning; ones keep all filters tied.

### Q 5.5.1 — Activation
**tanh: 96.94%** vs **ReLU: 97.83%** — slightly worse. tanh saturates and kills gradients for large inputs; ReLU does not for positives. Difference is small in shallow Net2.

### Q 5.6.1 — Batch size

| Batch size | Test accuracy |
|---|---|
| 8 | **97.98%** |
| 64 | **98.00%** |
| 128 | **98.04%** |

Small batches → noisier gradients, more updates/epoch, slower wall time. Large batches → smoother, more GPU-efficient. Accuracies nearly identical here.

### Q 5.7 — Architectures (bs=64)

| Model | Test accuracy |
|---|---|
| FCN (784→256→256→10) | **97.71%** |
| Net2 (1 conv) | **97.98%** |
| Net5 (2 conv + pool + large FC) | **99.02%** |

- **Best model:** Net5 — hierarchical spatial features + capacity.  
- **Convergence:** Net5 reaches high accuracy in fewer iterations; FCN is cheaper per step but needs more iterations. Faster convergence: tune LR/schedule, optimizer, batch size, init; add batch norm.  
- **Pooling:** downsamples (e.g. 28→14), reduces compute/parameters, adds shift robustness and larger receptive field.  
- **Loss vs accuracy:** yes — low loss can mean high confidence with more wrong top-1 guesses, or overfitting (low train loss, worse test accuracy).  
- **CNN vs FCN:** CNNs keep 2D structure, local connectivity, and weight sharing; FCNs flatten and ignore neighbors, so they need more parameters and relearn patterns at every location.

### Q 5.8 — Batch normalization
Net5 + BN after each conv and the hidden FC: **99.10%** vs **99.02%** without. BN stabilizes activations, allows stronger optimization, reduces sensitivity to init, and acts as light regularization. **Best Phase 1 model: Net5 + BatchNorm (99.10%).**

---

## Phase 2 — Transfer learning & augmentation on CIFAR-10

ResNet18 pretrained on ImageNet; final FC replaced for 10 classes. Training: SGD (lr 0.01, momentum 0.9, weight decay 5e-4), cosine schedule, **10 epochs**.

### Q 2.1.1 / 2.1.2 — Data augmentation
Augmentations create label-preserving variants (flips, crops, color changes) so the model generalizes and does not memorize pixels. Randomness means a new view each epoch.

| Augmentation | Role |
|---|---|
| Horizontal flip | Orientation invariance |
| Random crop (pad 4) | Position invariance |
| Color jitter | Lighting / camera robustness |
| Random resized crop | Scale / partial-object robustness |

### Q 2.2.1 — Custom rotation ±30°
Limits should keep labels valid and images natural. ±30° covers realistic camera/object tilt; larger angles look unnatural and add black corners. Tasks where orientation defines the class (e.g. 6 vs 9) need smaller ranges.

### Q 3.2.6 — With vs without augmentation

| Setup | Train acc | Test acc | Notes |
|---|---|---|---|
| Full fine-tune, no aug | 99.81% | **84.37%** | Overfit (test loss rose) |
| Full fine-tune + RandomCrop/HFlip | 89.08% | **85.87%** | Smaller train/test gap |

Augmentation made training harder but improved generalization (**+1.5%** test).

### Q 3.2.7 — Transfer learning
Reuse a model trained on a large dataset (ImageNet) as the start for a new task. Early layers already detect edges/textures/shapes, so training is faster, needs less data, and usually beats training from scratch.

### Q 3.2.8 — Freeze FC-only vs full fine-tune
FC-only: **42.61%** test; confusion matrix very diffuse (e.g. ship↔plane). Full fine-tune + aug: **85.87%**; mistakes mainly between similar classes (cat↔dog). Expected: ImageNet features at 224px do not match 32×32 CIFAR; a linear head cannot fix frozen features.

### Q 3.2.9 — When to freeze vs fine-tune
1. **Freeze all, train last layer** — small data, similar to ImageNet (e.g. few dog breeds).  
2. **Fine-tune entire net** — large or domain-shifted data (medical, satellite, low-res CIFAR).  
3. **Freeze convs, train all FC** — medium data, similar domain; need more classifier capacity than one linear layer.

### Q 3.2.11 — Partial unfreezing

| Trainable layers | Test accuracy | ~Time (10 epochs) |
|---|---|---|
| fc only | 42.61% | ~162 s |
| layer4 + fc | 70.05% | ~315 s |
| layer3 + layer4 + fc | **79.97%** | ~381 s |
| All layers + aug | **85.87%** | ~259 s |

Best partial freeze: **layer3 + layer4 + fc** (keep early edge/texture layers frozen). Full fine-tune still best because CIFAR is far from ImageNet resolution/domain.

---

## Phase 3 — Stanford Cars vehicle classifier (by make)

Open-ended make classifier (**49 classes**, e.g. Audi, BMW, Chevrolet) on the Stanford Cars dataset.

### Data
- Annotations from [jhpohovey/StanfordCars](https://github.com/jhpohovey/StanfordCars); makes parsed from class names (multi-word: AM General, Aston Martin, Land Rover).  
- Splits: **7,329 train / 815 val** (stratified 10% holdout) / **8,041 test**.  
- Resize to **224×224** (no center crop) so grille/badge/lights are kept.  
- Train aug: RandomResizedCrop (0.7–1.0), HorizontalFlip, TrivialAugmentWide, RandomErasing; ImageNet normalize.

### Model & training choices
| Choice | Value | Why |
|---|---|---|
| Backbone | ConvNeXt-Tiny (ImageNet-1k), all layers fine-tuned | Strong fine-grained backbone; Phase 2 showed full FT ≫ head-only |
| Head | Linear 768 → 49 | Matches make classes |
| Loss | Cross-entropy, label smoothing 0.1 | Less over-confident among near-identical makes |
| Optimizer | AdamW (backbone 1e-4, head 1e-3, wd 0.05) | Standard for ConvNeXt; larger LR for random head |
| Schedule | 1-epoch warmup + cosine, 20 epochs | Protect pretrained weights early; smooth anneal |
| Batch / other | 32; AMP; grad clip 1.0; best-val checkpoint | Fits GPU; stable; avoid last-epoch overfit |

### Results
- **Best validation accuracy: 97.18%** (epoch 15; train acc 98.10% that epoch).  
- **Test accuracy: 95.83%**.  
- Lowest per-class val recalls: Maybach (0.667), Scion / Geo (0.750) — rare classes.  
- Common confusions: Maybach→Lamborghini, Scion→Toyota, Tesla→BMW, GMC→Chevrolet, Cadillac→Chevrolet (shared platforms / design language).

### Fit analysis
Train and val loss fell while val accuracy rose to 97.18% then plateaued (~96–97%). Train/val gap stayed small (~98.6% vs ~96.8%), so the model **fits well** without clear overfitting. Train metrics are on heavy augmentation + label smoothing, so they understate clean-data fit.

### Possible improvements
Higher resolution (288–320), bbox crops, larger backbone, Mixup/CutMix, TTA (flip average), ensembling.

### Grading note
Section 7 of `Phase_3.ipynb` loads `vehicle_classifier.pth` standalone (paths for test images, `test_make.csv`, and weights). Weights are large and submitted via Canvas, not GitHub.

---

## Conclusions

1. **Phase 1:** Architecture and regularization matter most on MNIST — Net5 + BN reached **99.10%**; LR/optimizer/init also clearly affect learning quality.  
2. **Phase 2:** Pretrained ResNet18 needs **full fine-tuning + augmentation** for CIFAR-10 (**85.87%**); freezing early layers helps partially, but head-only transfer fails on this domain shift.  
3. **Phase 3:** Fine-grained make classification benefits from a modern ConvNet, strong augmentation, label smoothing, and differential LRs — **95.83%** test on 49 makes with a small train/val gap.

Overall, the lab progresses from building and diagnosing CNNs, through transfer learning and augmentation, to a competitive open-ended classifier that applies those lessons end-to-end.
