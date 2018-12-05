# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MAIN_DIR=/home/farbod/honours
MODEL=${MAIN_DIR}/exps/281018_alexnet_all_cub_lr005/checkpoint_100.pth.tar
ARCH='alexnet'
EXP=${MAIN_DIR}/exps/281018_alexnet_all_cub_lr005

MAIN_DIR=/home/farbod/honours/deepcluster
MODEL=${MAIN_DIR}/checkpoint.pth.tar
ARCH='alexnet'
EXP=${MAIN_DIR}/viz_alex

CONV=1

python gradient_ascent.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --arch ${ARCH}
