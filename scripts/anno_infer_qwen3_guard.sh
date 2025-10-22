#!/bin/bash
#SBATCH --job-name=classify          # 作业名称
#SBATCH --output=logs/classify_%j.out   # 标准输出日志 (%j会被作业ID替换)
#SBATCH --error=logs/classify_%j.err    # 错误输出日志
#SBATCH --partition=project
#SBATCH --account=cs707
#SBATCH --qos=cs707qos
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --cpus-per-task=8            # CPU核心数
#SBATCH --mem=64G                    # 内存
#SBATCH --time=24:00:00              # 最长运行时间

# 创建日志目录
mkdir -p logs

# 激活conda环境
source ~/.bashrc  # 或 source ~/miniconda3/etc/profile.d/conda.sh
conda activate med  # 替换为你的环境名

# 打印环境信息
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

# 运行你的脚本
python annotation/vllm_qwen3_guard_infer.py \
    --input /common/home/projectgrps/CS707/CS707G2/PKU-SafeRLHF/data/filter_pku_data.jsonl \
    --output /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno.jsonl

echo "Job finished at: $(date)"
