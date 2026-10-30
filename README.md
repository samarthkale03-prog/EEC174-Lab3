# EEC 174AY Lab 3: CNNs & Image Classification

**Students:** Samarth Kale, Saarth Kishore  

UC Davis, Fall 2026. This lab has three phases: building CNNs from scratch on MNIST, transfer learning with data augmentation on CIFAR-10, and an open-ended car-make classifier on the Stanford Cars dataset.

## Results at a glance

| Phase | Task | Best model | Test accuracy |
|---|---|---|---|
| 1 | MNIST digit classification | Net5 + Batch Norm | **99.10%** |
| 2 | CIFAR-10 transfer learning | ResNet18, fully fine-tuned + augmentation | **85.87%** |
| 3 | Stanford Cars, by make (49 classes) | ConvNeXt-Tiny, fully fine-tuned | **95.83%** |

## Repository contents

```
EEC174-Lab3/
├── Phase_1.ipynb   # CNNs on MNIST + hyperparameter experiments
├── Phase_2.ipynb   # Data augmentation + transfer learning on CIFAR-10
├── Phase_3.ipynb   # Stanford Cars vehicle classifier + report
├── README.md
└── .gitignore      # excludes data/, *.pth, stanford_cars/
```

Not included in the repo:
- **Datasets:** MNIST and CIFAR-10 download automatically into `data/`. Stanford Cars is obtained separately (see Phase 3).
- **`vehicle_classifier.pth`:** the trained Phase 3 weights (~110 MB, over GitHub's file size limit). They are submitted through Canvas.

## Phase 1: CNNs on MNIST

Builds and compares networks on 28×28 handwritten digits, with every experiment trained for 10 epochs.

| Experiment | Result |
|---|---|
| Learning rate | 3e-4 best (97.83%); 2e-3 overfit (97.78%); 5e-2 too high (88.94%) |
| Optimizer | Adam 97.83% > SGD momentum 0.9 97.44% > momentum 0.5 94.78% > SGD 93.42% |
| Weight init | Normal(0, 0.01) 98.09% > all ones 92.55% > all zeros 11.35% (no learning) |
| Activation | ReLU 97.83% > tanh 96.94% |
| Batch size | 128: 98.04%, 64: 98.00%, 8: 97.98% (8 is ~3× slower per epoch) |
| Architecture | Net5 99.02% > Net2 97.98% > FCN 97.71% |
| Batch norm | Net5 + BN 99.10%, with faster convergence |

**Models:**
- **Net2:** Conv(1→20, 3×3) → FC(13,520 → 10)
- **Net5:** Conv(1→32, 5×5) → MaxPool → Conv(32→64, 5×5) → FC(12,544 → 1024) → FC(1024 → 10)
- **FCN:** 784 → 256 → 256 → 10

## Phase 2: Transfer learning on CIFAR-10

ResNet18 pretrained on ImageNet, with the final layer replaced for 10 classes. Every run used SGD (lr 0.01, momentum 0.9, weight decay 5e-4) with a cosine schedule for 10 epochs.

| Setup | Test accuracy |
|---|---|
| Full fine-tune, no augmentation | 84.37% (99.8% train, so it overfit) |
| Full fine-tune + RandomCrop/HorizontalFlip | **85.87%** |
| Frozen, fc layer only | 42.61% |
| Train layer4 + fc | 70.05% |
| Train layer3 + layer4 + fc | 79.97% |

The notebook also covers the built-in augmentations (flip, crop, color jitter, random resized crop) and a custom `RandomRotate` (±30°) written with PIL only.

## Phase 3: Stanford Cars vehicle classifier

Classifies cars by **make** (49 classes, e.g. Audi, BMW, Chevrolet).

### Data
Data comes from the TA-approved repo [jhpohovey/StanfordCars](https://github.com/jhpohovey/StanfordCars):
```bash
git clone https://github.com/jhpohovey/StanfordCars.git
mv StanfordCars/stanford_cars ./stanford_cars
```
The first code cell in `Phase_3.ipynb` converts the repo's `.mat` annotations into `train_make.csv` and `test_make.csv`, taking the make from each class name (e.g. "Audi S5 Coupe 2012" → "Audi"). This gives 8,144 training images (7,329 train / 815 validation, stratified) and 8,041 test images.

### Model and training
| Choice | Value |
|---|---|
| Model | ConvNeXt-Tiny pretrained on ImageNet-1k, all layers fine-tuned |
| Input | 224×224, resized (not center-cropped) to keep the front and rear of each car |
| Augmentation | RandomResizedCrop, HorizontalFlip, TrivialAugmentWide, RandomErasing |
| Loss | Cross-entropy with label smoothing 0.1 |
| Optimizer | AdamW (backbone lr 1e-4, head lr 1e-3, weight decay 0.05) |
| Schedule | 1-epoch warmup + cosine decay, 20 epochs |
| Other | Mixed precision, gradient clipping, best-validation checkpoint |

### Results
- **Validation accuracy:** 97.18% (best epoch 15)
- **Test accuracy:** 95.83%
- **Most common confusions:** sister brands that share designs, e.g. Scion → Toyota, GMC → Chevrolet, Cadillac → Chevrolet
- **Fit:** validation loss decreased every epoch with a small train/validation gap (98.6% vs 96.8%), so there is no sign of overfitting

### Running the trained model (Section 7)
Section 7 of `Phase_3.ipynb` is standalone. Set the three paths at the top and run that cell alone:
```python
test_images_path = r'path/to/cars_test'
test_annotations_path = r'path/to/test_make.csv'
weights_path = r'path/to/vehicle_classifier.pth'
```
It defines `CarsDataset`, loads the model as `vehicle_classifier` (the class names are stored in the checkpoint), and prints the test accuracy.

## Environment

- Windows 11, VS Code + Jupyter
- Python 3.13 virtual environment
- PyTorch 2.14 (CUDA 12.6), torchvision
- scikit-learn, pandas, matplotlib, scipy
- NVIDIA RTX 4050 Laptop GPU (6 GB)

Setup:
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install scikit-learn pandas matplotlib scipy ipykernel
```

## References

- Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). *3D Object Representations for Fine-Grained Categorization.* ICCV Workshops.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s.* CVPR.
