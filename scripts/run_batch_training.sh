#!/bin/bash

# Mini-batch训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 参数设置
DATA_DIR="./data"
CONFIG="./config/default_config.yaml"
EXP_NAME="moe_kgc_batch_experiment"
TASK="link_prediction"
EPOCHS=100
BATCH_SIZE=32
LR=0.001

# Mini-batch参数
NUM_NEIGHBORS="25 10"
SAMPLING_METHOD="neighbor"

# 创建实验目录
mkdir -p experiments/${EXP_NAME}/{checkpoints,logs}

# 第一步：预处理数据为PyG格式（如果需要）
if [ ! -f "${DATA_DIR}/train/pyg_data.pt" ]; then
    echo "预处理数据为PyG格式..."
    python scripts/prepare_pyg_data.py \
        --data_dir ${DATA_DIR} \
        --config ${CONFIG}
fi

# 第二步：运行训练
echo "开始mini-batch训练..."
python scripts/train_e2e.py \
    --data_dir ${DATA_DIR} \
    --config ${CONFIG} \
    --task ${TASK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --exp_name ${EXP_NAME} \
    2>&1 | tee experiments/${EXP_NAME}/logs/training.log

# 第三步：评估
echo "评估模型..."
python scripts/train_e2e.py \
    --data_dir ${DATA_DIR} \
    --config ${CONFIG} \
    --task ${TASK} \
    --checkpoint experiments/${EXP_NAME}/checkpoints/best_model.pt \
    --exp_name ${EXP_NAME}_eval \
    --eval_only \
    2>&1 | tee experiments/${EXP_NAME}/logs/evaluation.log

echo "训练完成！结果保存在 experiments/${EXP_NAME}"