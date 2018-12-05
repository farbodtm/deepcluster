# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MAIN_DIR=/home/farbod/honours
DATA=${MAIN_DIR}/datasets/miniimagenet/
#MODEL="${MAIN_DIR}/exps/251118_alexnet_mimagenet_last_layer/checkpoints/checkpoint_0.pth.tar"
MODEL="${MAIN_DIR}/deepcluster/checkpoint.pth.tar"
EXP="${MAIN_DIR}/exps/291118_alexnet_mimagenet_last_layer"


mkdir -p ${EXP}

python eval_linear.py --model ${MODEL} --data ${DATA} --conv 5 --lr 0.01 \
  --wd -6 --verbose --exp ${EXP} --workers 4 --batch_size 128
