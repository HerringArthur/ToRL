#!/bin/bash

# ================================================================
# TORL (Tool-Integrated Reinforcement Learning) 训练启动脚本
# ================================================================

# 1. 环境设置
# ----------------------------------------------------------------
# 指定使用的 GPU 编号 (例如 "0" 或 "0,1")
export CUDA_VISIBLE_DEVICES=0

# 设置 Tokenizers 并行度，防止死锁警告
export TOKENIZERS_PARALLELISM=false

# 2. 路径配置
# ----------------------------------------------------------------
# 基础模型路径 (可以是 HuggingFace ID 或 本地绝对路径)
# 建议尝试: "Qwen/Qwen2.5-7B-Instruct" 或 "Qwen/Qwen2.5-1.5B-Instruct" (显存更小)
MODEL_PATH="/seu_share2/home/fenglei/sharedata/Qwen2.5-1.5B-Instruct"

# 数据集路径
DATA_PATH="/seu_share2/home/fenglei/213243847/data/grpo-aco/data/rl_dataset_llm_v2.json"

# 输出保存路径
OUTPUT_DIR="./output/checkpoints_verl_v1"

# 3. 核心超参数配置 (根据显存大小调整)
# ----------------------------------------------------------------
# GRPO 组采样大小 (显存占用大户)
# - 24GB 显存 (3090/4090): 建议设为 4 ~ 8
# - 40GB+ 显存 (A100): 建议设为 16
GROUP_SIZE=8

# 学习率 (RL 阶段通常需要很小的 LR)
LEARNING_RATE=1e-6

# 训练轮数
NUM_EPOCHS=3

# 奖励权重 (生成正确 JSON 文件且内容正确的奖励)
SUCCESS_REWARD=2.0

# LoRA Rank (秩)，越大参数越多，通常 16-64 之间
LORA_R=16

# 4. 启动训练命令
# ----------------------------------------------------------------
echo "Starting TORL training..."
echo "Model: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Group Size: $GROUP_SIZE"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 记录日志到 train.log
python train_torl.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --group_size $GROUP_SIZE \
    --max_steps_per_episode 15 \
    --save_every 50 \
    --lora_r $LORA_R \
    --reward_success_weight $SUCCESS_REWARD \
    --use_lora \
    --use_amp \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training finished. Logs saved to $OUTPUT_DIR/train.log"