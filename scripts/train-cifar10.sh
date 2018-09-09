#!/usr/bin/env bash
CFG=$1
echo "CFG=${CFG}"
shift

GPU=$1
echo "GPU=${GPU}"
shift

for SEED in $@
do
    echo "SEED=${SEED}"
    CUDA_VISIBLE_DEVICES=$GPU python3 train_cifar10.py -c $CFG --rng-seed $SEED
done
