# Neural Memory Ordinary Differential Equation Networks: Formulation, Mechanisms, and Applications (AAAI 2025)

This is the source code for the AAAI 2025 paper.

# Evaluating Long-Term Memory
The source code is under `Image Classification` folder.

For albation, the source code is under `Image Classification/ablation` folder.

* *Models*: nmODE
* *Metrics*: ACC
* *Data sets*: Fashion-MNIST

For comparison, the source code is under `Image Classification/compare` folder.

* *Models*: ResNet+nmODE vs ResNet, DenseNet+nmODE vs DenseNet, VGG+nmODE vs VGG
* *Metrics*: ACC
* *Data sets*: CIFAR-10, CUB-2011-200

# Assessing Extended Sequences Modeling
The source code is under `LSTF` folder.

* *Models*: nmODE vs SpaceTime, NLinear, FILM, S4, FedFormer, Autoformer, Informer, ARIMA
* *Metrics*: MSE, MAE
* *Data sets*: ETT

For comparison, run:
```
python main.py
```

# Analyzing Computational Efficiency
The source code is under `Speed` folder.

* *Models*: parallel nmODE, non-parallel nmODE
* *Metrics*: forward and backward speed (s)

For evaluate, run:
```
python speed.py
```