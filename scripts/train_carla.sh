#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for training on the LLFF dataset.

SCENE=25_box
EXPERIMENT=Carla_dend_cntr02_0_200_8_256_10fipe_gtbox_noboxcntr
TRAIN_DIR=/home/tristram/nerf_results/$EXPERIMENT/$SCENE
DATA_DIR=/home/tristram/data/carla/$SCENE/


rm $TRAIN_DIR/*
python -m train_boxpose \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=configs/carla_dyn.gin \
  --chunk=1024 \
  --logtostderr