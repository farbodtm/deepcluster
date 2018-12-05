#!/bin/bash

MAIN_DIR=/home/farbod/honours

#DATASET_PATH=${MAIN_DIR}/products/Stanford_Online_Products
#DATASET_PATH=${MAIN_DIR}/cub-200-2011/CUB_200_2011/train
DATASET_PATH=${MAIN_DIR}/datasets/miniimagenet/train
#DATASET_PATH=${MAIN_DIR}/cub-200-2011/CUB_200_2011/images
#DATASET_PATH=/media/robotvision2/H/vijay_imagenet/train/
#EXP=${MAIN_DIR}/exps/281018_alexnet_all_cub_lr005
EXP=${MAIN_DIR}/exps/251118_alexnet_mimagenet_clustering_test

mkdir -p ${EXP}
#--resume ${EXP_DIR}/checkpoint-150.pth.tar
python main.py --exp ${EXP} --verbose --lr 0.02 --nmb_cluster 65 --arch alexnet --batch 64 --sobel $DATASET_PATH

