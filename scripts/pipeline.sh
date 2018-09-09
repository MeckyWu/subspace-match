#!/usr/bin/env bash
MODEL=$1

# train
bash scripts/train-cifar10.sh $MODEL 0,1 0 1 2

# predicate
bash scripts/pred-cifar10.sh $MODEL relu 0 1 2

# calculate maximal match
for EPS in 0.15 0.2 0.25 0.3 0.5
do
    bash scripts/max-match-cifar10.sh $MODEL $EPS 0 1 2
done
