#!/bin/sh
set -xe
if [ ! -f main.py ]; then
    echo "Please make sure you run this from the project's top level directory."
    exit 1
fi;

python3 -u main.py \
  --epochs 100 \
  --P 0.75 \
  --C 78 \
  --K 4 \
  --M 1 \
  --seed 0 \
  --graph-mode 'WS' \
  --N 32 \
  --stages 3 \
  --learning-rate 0.1 \
  --batch-size 100 \
  --regime 'small' \
  --dataset 'CIFAR100' \
  --augmented 'False' \
