Official pytorch implementation of the paper: 

**"Accelerating Convergence in Bayesian Few-Shot Classification"**

Requirements
-------------

1. Python >= 3.x
2. Numpy >= 1.17
3. [pyTorch](https://pytorch.org/) >= 1.2.0
4. [GPyTorch](https://gpytorch.ai/) >= 0.3.5
5. (optional) [TensorboardX](https://pypi.org/project/tensorboardX/) 
 
Installation
-------------

```
pip install numpy torch torchvision gpytorch h5py pillow
```

We confirm that the following configuration worked for us: numpy 1.22.4, torch 1.11.0, torchvision 0.12.0, gpytorch 1.9.0, h5py 3.4.0, pillow 9.1.1

MD-BSFC: code of our method
--------------------------

**Classification.** The code for the classification case is accessible in [MD.py](./methods/MD.py), with most of the important pieces contained in the `train_loop()` method (training), and in the `correct()` method (testing). 

Experiments
============

These are the instructions to train and test the methods reported in the paper in the various conditions.

**Download and prepare a dataset.** This is an example of how to download and prepare a dataset for training/testing. Here we assume the current directory is the project root folder:

```
cd filelists/DATASET_NAME/
sh download_DATASET_NAME.sh
```

Replace `DATASET_NAME` with one of the following: `CUB`, `miniImagenet`, `omniglot`, `emnist`. Notice that the link to mini-Imagenet is no longer available on the official website. We refer to [mini-imagenet-tools](https://github.com/yaoyao-liu/mini-imagenet-tools) for various alternatives to downloading the dataset. 

Classification
---------------

**Train classification.** Our methods can be trained using the following syntax:

```
python train.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
```

This will train MD 5-way 1-shot on the CUB dataset with seed 1 and the ELBO loss. The `dataset` string can be one of the following: `CUB`, `miniImagenet`, `cross`, `cross_char`. At training time the best model is evaluated on the validation set and stored as `best_model.tar` in the folder `./save/checkpoints/DATASET_NAME`. The parameter `--train_aug` enables data augmentation. The parameter `seed` set the seed for pytorch, numpy, and random. Set `--seed=0` or remove the parameter for a random seed. The parameter `steps` controls the task-level update steps for inner-loop update steps of mirror descent. The parameter `loss` can be `ELBO` or `PL`, corresponding to the ELBO loss and the predictive likelihood loss (only used in converging experiments). Additional parameters are provided in the file `io_utils.py`, such as `mean` and `kernel`, where `mean` sets the prior mean for GP (default is 0) and the choices of kernel include `linear`, `rbf`,  `matern`, `poli1`, `poli2`, `bncossim`.

**Test classification.** For testing our methods it is enough to repeat the train command replacing the call to `train.py` with the call to `test.py` as follows:

```
python test.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
```

**Calibration.** For calibrating our methods it is enough to repeat the train command replacing the call to `train.py` with the call to `calibrate.py` as follows:

```
python calibrate.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=53 --train_aug --steps=3 --tau=1 --loss="ELBO"
```

**Reproduction of our results.** You can simply run

```
sh classification.sh
```

and

```
sh calibrate.sh
```

to reproduce the results presented in the paper.

<!-- Acknowledgements
---------------

This repository is a fork of [https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer). -->
