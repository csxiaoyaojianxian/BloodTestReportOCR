#!/bin/bash
set -e

paddle train \
  --config=trainer_config_age.py \
  --save_dir=./output_age \
  --trainer_count=1 \
  --num_passes=100 \
  --use_gpu=1 \
  2>&1 | tee 'train_age.log'
