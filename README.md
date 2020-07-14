# Subspace Inference for Bayesian Deep Learning

This repository contains a PyTorch code for the subspace inference method introduced in the paper

[Subspace Inference for Bayesian Deep Learning](https://arxiv.org/abs/1907.07504)

by [Pavel Izmailov](https://izmailovpavel.github.io/), [Wesley Maddox](https://wjmaddox.github.io/), [Polina Kirichenko](https://github.com/PolinaKirichenko), [Timur Garipov](https://github.com/timgaripov), [Dmitry Vetrov](https://bayesgroup.ru/), and [Andrew Gordon Wilson](https://people.orie.cornell.edu/andrew/)

## Introduction

For deep neural network models, exact Bayesian inference is intractable, and existing approximate inference methods suffer from many limitations, largely due to the high dimensionality of the parameter space.
In subspace inference we address this problem by performing inference in low-dimensional subspaces of the parameter space. 
In the paper, we show how to construct very low-dimensional subspaces that can contain diverse high performing models. 
Bayesian inference within such subspaces is much easier than in the original parameter space and leads to strong results.  

At a high level, our method proceeds as follows: 
1. We construct a low dimensional subspace of the parameter space
2. We approximate the posterior distribution over the parameters in this subspace
3. We sample from the approximate posterior in the subspace and perform Bayesian model averaging.

<!--
<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/60686210-49921100-9e5c-11e9-8625-532951f69c1f.png" height=275>
  <img src="https://user-images.githubusercontent.com/14368801/60686208-49921100-9e5c-11e9-9968-f7d9cfdaf8a0.png" height=275>
</p>
-->

<p align="center">
  <img src=https://user-images.githubusercontent.com/14368801/60688558-ccba6380-9e6a-11e9-852e-2df28bed04db.gif height=300>
</p>
  
<!--
<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/60686209-49921100-9e5c-11e9-9b74-a98497dfc7d8.png" height=275>
  <img src="https://user-images.githubusercontent.com/14368801/60686207-49921100-9e5c-11e9-8ed7-e5c684597edb.png" height=275>
</p>
-->
  
For example, we can achieve state-of-the-art results for a WideResNet with 36 million parameters by performing approximate inference in only a 5-dimensional subspace.

If you use our work, please cite the following paper:
```
@article{izmailov_subspace_2019,
  title={Subspace Inference for Bayesian Deep Learning},
  author={Izmailov, Pavel and Maddox, Wesley and Kirichenko, Polina and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={Uncertainty in Artificial Intelligence (UAI)},
  year={2019}
}
```

## Installation

You can install the package by running the following command
```bash
python setup.py
```

## Usage

We provide the scripts and example commands to reproduce the experiments from the paper.

### Subspace Construction and Inference Procedures

In the paper we show how to construct random, PCA, and mode-connected subspaces, over which we can perform Bayesian inference. The PCA subspace will often be most practical, in terms of a runtime-accuracy trade-off. Once the subspace is constructed, we consider various approaches to inference within the subspace, such as MCMC and variational methods. The following examples show how to call subspace inference with your choice of subspace and inference procedure.

### Visualizing Regression Uncertainty

We provide a [jupyter notebook](experiments/synthetic_regression/visualizing_uncertainty.ipynb) in which we show how to apply subspace inference to a regression problem, using different types of subspaces and approximate inference methods.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/60686210-49921100-9e5c-11e9-8625-532951f69c1f.png" height=150>
  <img src="https://user-images.githubusercontent.com/14368801/60686208-49921100-9e5c-11e9-9968-f7d9cfdaf8a0.png" height=150>
  <img src="https://user-images.githubusercontent.com/14368801/60686209-49921100-9e5c-11e9-9b74-a98497dfc7d8.png" height=150>
  <img src="https://user-images.githubusercontent.com/14368801/60686207-49921100-9e5c-11e9-8ed7-e5c684597edb.png" height=150>
</p>

### Image Classification

The folder [`experiments/cifar_exps`](experiments/cifar_exps) contains the implementation of subspace inference for CIFAR-10 and CIFAR-100 datasets. 
To run subspace inference, you need to first pre-train the subspace, and then either run variational inference (VI) or elliptical slice sampling (ESS) in the subspace.
For example, you can pre-train a PCA subspace for a VGG-16 by running
```
python3 experiments/cifar_exps/swag.py --data_path=data --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test \
      --dir=ckpts/vgg16_run1
```
Then, you can run ESS in the constructed subspace using the following command
```
python3 experiments/cifar_exps/subspace_ess.py --dir=ckpts/vgg16_run1/ess/ --dataset=CIFAR100 \
      --data_path=~/datasets/ --model=VGG16 --use_test --rank=5 --checkpoint=ckpts/vgg16_run1/swag-300.pt \
      --temperature=5000 --prior_std=2
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/60690876-db5f4580-9e7f-11e9-8d6f-0e7fb8f5c11f.png" height=200>
  <img src="https://user-images.githubusercontent.com/14368801/60690904-0b0e4d80-9e80-11e9-8713-29bd3b6b1397.png" height=200>
  <img src="https://user-images.githubusercontent.com/14368801/60690903-0b0e4d80-9e80-11e9-9357-aff77306f4fc.png" height=200>
</p>

Please refer to the [`README`](experiments/cifar_exps/README.md) for more detailed information.

### UCI Regression

The folder [`experiments/uci_exps`](experiments/uci_exps) contains implementations of the subspace inference procedure 
for UCI regression problems.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/60690931-6b04f400-9e80-11e9-995b-d04ec9feaa9b.png" height=150>
</p>

Please refer to the [`README`](experiments/uci_exps/README.md) for more detailed information.

## References for Code Base

Repositories: 
  - Stochastic weight averaging (SWA): [PyTorch repo](https://github.com/timgaripov/swa/). Many of the base methods and model definitions are built off of this repo.
  - SWA-Gaussian (SWAG): [PyTorch repo](https://github.com/wjmaddox/swa_gaussian).
  our codebase here is based the SWAG repo; SWAG repo also contains the implementations of various baselines fof 
  image classification as well as the experiments on Hessian eigenvalues.
  - Mode-Connectivity: [PyTorch repo](https://github.com/timgaripov/dnn-mode-connectivity). 
  We use the scripts from this repository to construct our curve subspaces.
  - Bayesian benchmarks: [Repo](https://github.com/hughsalimbeni/bayesian-benchmarks); the `experiments/uci_exps` folder is a clone of that repo.

Model implementations:
  - VGG: https://github.com/pytorch/vision/
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - WideResNet: https://github.com/meliketoy/wide-resnet.pytorch

