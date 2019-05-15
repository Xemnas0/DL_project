#!/bin/sh
set -xe
if [ ! -f main.py ]; then
    echo "Please make sure you run this from the project's top level directory."
    exit 1
fi;

python3 -u main.py \
  --epochs 100 \
  --P 0.2 \
  --C 78 \
  --K 4 \
  --M 1 \
  --seed 0 \
  --graph-mode 'ER' \
  --N 32 \
  --stages 2 \
  --learning-rate 0.1 \
  --batch-size 128 \
  --regime 'small' \
  --dataset 'CIFAR100' \
  --augmented 1 \
  --decay 5e-9 \
  --stride 1 \
  --lr_period 50 \
  --min_lr 1e-6 \
  --update_type_lr 'batch'