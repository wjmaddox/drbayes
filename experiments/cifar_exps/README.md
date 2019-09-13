# Image Classification

The scripts in `experiments/cifar_exps/run_swag.py` allow to run subspace inference on CIFAR-10 and CIFAR-100 datasets.


## Subspace Construction

To run subspace inference we need to first construct the subspace.
In order to construct a PCA or random subspace, we train a SWAG solution 
using the following command (code adapted from https://github.com/wjmaddox):

```
python experiments/cifar_exps/swag.py \
      --dir=<DIR> \
      --dataset=<DATASET> \
      --data_path=<PATH> \
      --model=<MODEL> \
      --epochs=<EPOCHS> \
      --lr_init=<LR_INIT> \
      --wd=<WD> \
      --swag \
      --cov_mat \
      --swag_start=<SWAG_START> \
      --swag_lr=<SWAG_LR> \
      --max_num_models=<RANK> \
      --seed=<SEED> \
      [--use_test]
```
Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16
    - PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWAG_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWAG_LR``` &mdash;  SWA learning rate (default: 0.05)
* ```RANK``` &mdash; rank of the low-rank part of the SWAG covariance approximation; this is an upper-bound on the dimension of the PCA subspace that can be extracted from the SWAG solution; (default: 20). 
* ```SEED``` &mdash; random seed. 
* ```--use_test``` &mdash; use test data to evaluate the method; by default validation data is used for evaluation. 

We list the scripts that were used to pretrain subspaces for each of the datasets and architectures below.
Note that the hyperparameters are directly adapted from [SWAG](https://github.com/wjmaddox).

```bash
# PreResNet164, CIFAR100
python3 experiments/cifar_exps/swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swag --swag_start=161 --swag_lr=0.05 --cov_mat --use_test \
      --dir=<DIR>

# PreResNet164, CIFAR10
python experiments/cifar_exps/swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR10 --save_freq=300 \  
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test \
      --dir=<DIR>
      
# WideResNet28x10:
python experiments/cifar_exps/swag.py --data_path=<PATH> --epochs=300 --dataset=[CIFAR10 or CIFAR100] --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.05 --cov_mat --use_test \
      --dir=<DIR>

# VGG16:
python experiments/cifar_exps/swag.py --data_path=<PATH> --epochs=300 --dataset=[CIFAR10 or CIFAR100] --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swag --swag_start=161 --swag_lr=0.01 --cov_mat --use_test \
      --dir=<DIR>
```

The SWAG checkpoints contain sufficient information to construct both random and PCA subspaces.

### Curve Subspace

Constructing the curve subspace requires several additional steps. 
First, you need to train two SWAG solutions [as above](README.md#subspace-construction) with different values of the random seed (`--seed` parameter).
Running `experiments/cifar_exps/swag.py` will produce the SWA checkpoint at `<DIR>/swa.pt`

Then, you need to run the code from the [mode connectivity repo](https://github.com/timgaripov/dnn-mode-connectivity)
to connect the find a low-loss curve connecting the two swa solutions. 
You can reuse the hyper-parameters provided [here](../dnn-mode-connectivity/ckpts/c100/preresnet164/curve2/checkpoint-600.pt) and just need to substitute the checkpoints for the `swa.pt` checkpoints constructed at the previous step.
We also recommend setting `--epochs=600` for all architectures to ensure convergence.
At the end of this step you should have a checkpoint `checkpoint-600.pt`, that contains information required to conctruct the curve subspace.

## Approximate Inference within a Subspace

Once you have the checkpoints containing the SWAG solution or a mode-connecting curve, you can run approximate inference within the corresponding subspace. 

## Elliptical Slice Sampling

To run ESS in a subspace use the following command:
```
python experiments/cifar_exps/subspace_ess.py \
      --dir=<DIR> \
      --dataset=<DATASET> \
      --data_path=<PATH> \
      --model=<MODEL> \
      --wd=<WD> \
      --num_samples=<NUM_SAMPLES> \
      --checkpoint=<CKPT> \
      --rank=<RANK> \
      --temperature=<TEMPERATURE> \
      --prior_STD=<PRIOR_STD>
      [--use_test \]
      [--random \]
      [--curve]
```
Parameters:

* ```NUM_SAMPLES``` &mdash; number of samples ESS will produce (default: 30)
* ```RANK``` &mdash; dimension of the subspace; for curve subspace it must be equal to `2`, and 
for PCA subspace it can't be larger than `RANK` used in `swag.py` (default: 2)
* ```CKPT``` &mdash; path to the checkpoint produced by `swag.py` or mode-connectivity script
* ```TEMERATURE``` &mdash; value of the temperature parameter of subspace inference (default: 1)
* ```PRIOR_STD``` &mdash; prior standard deviation (default: 1)
* ```--random``` &mdash; use random subspace; by default PCA subspace is used.
* ```--curve``` &mdash; use curve subspace; by default PCA subspace is used.

The other hyperparameters are the same as in the [`experiments/cifar_exps/swag.py`](README.md#subspace-construction) script.

For example, to run ESS on PreResNet-164 on CIFAR-100 in a curve subspace stored in `ckpts/curve.pt` you can use the following command
```
python3 experiments/cifar_exps/subspace_ess.py --dir=ckpts/curve_ess/ --dataset=CIFAR100 --data_path=data \
      --model=PreResNet164 --use_test --curve --checkpoint=ckpts/curve.pt --temperature=10000 --prior_std=1
```

## Variational Inference

Alternatively, you can run VI in the subspace using the command
```
python experiments/cifar_exps/subspace_vi.py \
      --dir=<DIR> \
      --dataset=<DATASET> \
      --data_path=<PATH> \
      --model=<MODEL> \
      --wd=<WD> \
      --num_samples=<NUM_SAMPLES> \
      --checkpoint=<CKPT> \
      --rank=<RANK> \
      --temperature=<TEMPERATURE> \
      --lr=<LR> \
      --prior_std=<PRIOR_STD> \
      --init_std=<INIT_STD> \
      [--use_test \]
      [--no_mu \]
      [--random]
```
* ```LR``` &mdash; learning rate for variational inference (default: 30)
* ```INIT_STD``` &mdash; initial value of the standard deviations in the variational approximation (default: 30)
* ```--no_mu``` &mdash; if specified, the mean is fixed to zero in the variational approximation
See [`experiments/cifar_exps/subspace_ess.py`](README.md#elliptical-slice-sampling) for a descreption of the other parameters. Note, that you can't use `--curve` with `subspace_vi.py`.

For example, to run VI on a VGG-16 on CIFAR-100 in the PCA  subspace stored in `ckpts/swag-300.pt`, use the following command
```
python3 experiments/cifar_exps/subspace_vi.py --data_path=~/datasets/ --epochs=30 --num_samples=30 --dataset=CIFAR100 \
      --model=VGG16 --rank=5 --max_rank=20 --use_test  --dir=ckpts/vi/ --checkpoint=ckpts/swag-300.pt \
      --temperature=5000 --no_mu
```

## Results

In the table below we present the negative log likelihoods (NLL) and accuracy for ESS in different subspaces for PreResNet-164 on CIFAR-100 datasets.
Please see the paper for more detailed results.
      
| PreResNet-164, CIFAR-100  |  Random         | PCA            | Curve         | 
| ------------------------- |:---------------:|:--------------:|:-------------:|
| NLL                       | 0.6858 ± 0.0052 | 0.6652 ± 0.004 | 0.6493 ± 0.01 |  
| Accuracy                  | 80.17 ± 0.03    | 80.54 ± 0.13   | 81.55 ± 0.26  |

## Baselines

See [SWAG repo](https://github.com/wjmaddox/swa_gaussian) for implementations of SWAG and other baselines.
