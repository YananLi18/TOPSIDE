#!/bin/sh

export GPU_ID=$1

if [ -n "$2" ]; then
    # 如果存在，则将变量设置为第二个参数的值
    export LOSS_NAME=$2
else
    # 如果不存在，可以设置为默认值或保持不变
    export LOSS_NAME='5'
fi



echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python train_maml_system.py --name_of_args_json_file experiment_config/sneset_maml++-sneset_100_8_0.1_48_5_0.json --gpu_to_use $GPU_ID --num_classes_per_set $LOSS_NAME
