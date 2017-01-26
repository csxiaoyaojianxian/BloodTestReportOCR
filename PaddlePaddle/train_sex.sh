#!/bin/bash
set -e

paddle train \
  --config=trainer_config_sex.py \
  --save_dir=./output_sex \
  --trainer_count=1 \
  --num_passes=30 \
  --use_gpu=1 \
  2>&1 | tee 'train_sex.log'
