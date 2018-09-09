# To What Extent Do Different Neural Networks Learn the Same Representation: A Neuron Activation Subspace Match Approach

## Requirements
- python3
- anaconda
- pytorch

A easy way is to run the following commands after installing conda
```
conda create -n pytorch python3.6 anaconda
conda install pytorch torchvision -c pytorch
source activate pytorch
```

## Project Structure
**train_cifar10.py** is the main script to train or validate on cifar10.

**configs/cifar10** contains configurations for training models, like **vgg16.py**.

You could run `python train_cifar10.py -c vgg16 --rng_seed 0` to train a vgg16 model on cifar10 with random seed 0 with all available gpus.

Results would be saved at **output/cifar10/{config}/{rng_seed}**.

All the models are defined in **models/cifar10**.

**pred_cifar10.py** is the main script to extract features on cifar10
```
# select all activated features and output a compressed npz on test set
python pred_cifar10.py -m vgg16 --ckpt output/cifar10/vgg16/rnd_0/best.ckpt -t relu
```

**scripts** maintains some useful scripts for convenience.

## Todo
- [ ] Test training scripts by visualization
- [ ] Test feature extractors
- [ ] Test maximal matching calculation
- [ ] Test minimum matching calculation
- [ ] Add docstrings, copyright, license
