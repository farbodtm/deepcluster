#!/bin/bash

MAIN_DIR=/home/farbod/honours

#DATASET_PATH=/media/robotvision2/H/vijay/code_triplet/evaluation/evaluation_dag_final/IMAGES_LABELS_BIRDS_TEST.mat
DATASET_PATH=${MAIN_DIR}/datasets/CUB_200_2011/test.mat
WEIGHTS=${MAIN_DIR}/exps/291118_inceptionv1_mimagenet_test/checkpoint_10.pth.tar
#WEIGHTS=${MAIN_DIR}/exps/231018_inceptionv1_freeze/checkpoint_200.pth.tar
#WEIGHTS=${MAIN_DIR}/exps/221018_alexnet/checkpoint.pth.tar

python features.py --weights ${WEIGHTS} --verbose --arch inceptionv1 $DATASET_PATH
