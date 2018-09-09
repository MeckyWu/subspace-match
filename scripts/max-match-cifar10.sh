#!/usr/bin/env bash
MODEL=$1
echo "MODEL=${MODEL}"
shift

EPSILON=$1
echo "EPSILON=${EPSILON}"
shift

for S0 in $@
do
    shift
    for S1 in $@
    do
        echo "S0=${S0}  S1=${S1}"
        python3 calc_max_match.py \
            -e $EPSILON \
            --mat-dir output/cifar10_matrix/$MODEL \
            --s0 $S0 \
            --s1 $S1
    done
done