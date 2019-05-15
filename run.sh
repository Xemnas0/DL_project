#!/bin/sh
set -xe
if [ ! -f main.py ]; then
    echo "Please make sure you run this from the project's top level directory."
    exit 1
fi;

python3 -u main.py \
  --epochs 100 \
  --P 0.75 \
  --C 64 \
  --K 4 \
  --M 1 \
  --seed 0 \
  --graph-mode 'WS' \
  --N 32 \
  --stages 2 \
  --learning-rate 0.1 \
  --batch-size 128 \
  --regime 'small' \
  --dataset 'CIFAR10' \
  --augmented 1 \
  --decay 5e-9 \
  --stride 1