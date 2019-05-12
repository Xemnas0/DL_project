#!/bin/sh
set -xe
if [ ! -f main.py ]; then
    echo "Please make sure you run this from the project's top level directory."
    exit 1
fi;

python3 -u main.py \
--epochs 5 \
--P 0.75 \
--C 32 \
--K 4 \
--M 1 \
--seed 0 \
--graph-mode WS \
--N 32 \
--stages 1 \
--learning-rate 1e-2 \
--batch-size 32 \
--regime 'small' \
--dataset 'MNIST' \
--distributed 0 \