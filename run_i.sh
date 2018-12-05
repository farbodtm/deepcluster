#!/bin/bash

MAIN_DIR=/home/farbod/honours

#DATASET_PATH=${MAIN_DIR}/products/Stanford_Online_Products
DATASET_PATH=${MAIN_DIR}/datasets/CUB_200_2011/train
#DATASET_PATH=${MAIN_DIR}/cub-200-2011/CUB_200_2011/train
#DATASET_PATH=${MAIN_DIR}/datasets/miniimagenet/train
#DATASET_PATH=/media/robotvision2/H/vijay_imagenet/train/
#EXP=${MAIN_DIR}/exps/281018_alexnet_all_cub_lr005
EXP=${MAIN_DIR}/exps/291118_inceptionv1_mimagenet_test2

mkdir -p ${EXP}
#--resume ${EXP_DIR}/checkpoint-150.pth.tar
python main_i.py --exp ${EXP} --verbose --lr 0.01 --num_classes 100 --arch inceptionv1 --batch 64 --epochs 100 $DATASET_PATH 
#python main_i.py --exp ${EXP} --verbose --lr 0.001 --num_classes 65 --arch alexnet --batch 128 --epochs 100  $DATASET_PATH 
