#!/bin/bash

MAIN_DIR=/home/farbod/honours

DATASET_PATH=${MAIN_DIR}/datasets/miniimagenet/train
VAL_PATH=${MAIN_DIR}/datasets/miniimagenet/val
EXP=${MAIN_DIR}/test/mnist

mkdir -p ${EXP}

python imagenet.py --exp ${EXP} --wd 1e-4 --epochs 10 --lr 0.01 --arch alexnet --batch 64 --train-dir $DATASET_PATH --val-dir $VAL_PATH
