#!/bin/bash

export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=0
export base_output_dir="./cola"

for lora_r in 32 64 128 256; do
  for ending_step_ratio in .4 .6 .8; do
    output_dir="$base_output_dir/lora_r${lora_r}_ending${ending_step_ratio}"

    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
    examples/text-classification/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name cola \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 64 \
    --learning_rate 4e-4 \
    --num_train_epochs 80 \
    --output_dir $output_dir/model \
    --overwrite_output_dir \
    --logging_steps 10 \
    --ending_step_ratio $ending_step_ratio \
    --logging_dir $output_dir/log \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --warmup_ratio 0.06 \
    --apply_lora \
    --lora_r $lora_r \
    --lora_alpha 16 \
    --seed 0 \
    --weight_decay 0.1
  done
done