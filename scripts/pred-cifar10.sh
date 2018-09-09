#!/usr/bin/env bash
MODEL=$1
echo "MODEL=${MODEL}"
shift

TYPE=$1
echo "TYPE=${TYPE}"
shift

GPU=0

for SEED in $@
do
    CKPT_PATH="./output/cifar10/${MODEL}/rnd_${SEED}/best.ckpt"
    echo $CKPT_PATH
    CUDA_VISIBLE_DEVICES=$GPU python3 pred_cifar10.py --ckpt $CKPT_PATH -m $MODEL -t $TYPE
done
