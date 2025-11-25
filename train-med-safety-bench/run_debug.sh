#!/bin/bash
#SBATCH --job-name=debug          # 作业名称
#SBATCH --output=logs/debug_%j.out   # 标准输出日志 (%j会被作业ID替换)
#SBATCH --error=logs/debug_%j.err    # 错误输出日志
#SBATCH --partition=project
#SBATCH --account=cs707
#SBATCH --qos=cs707qos
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --cpus-per-task=8            # CPU核心数
#SBATCH --mem=64G                    # 内存
#SBATCH --time=24:00:00              # 最长运行时间

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# GPU信息
echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "=========================================="
echo "Starting Debug Inference..."
echo "=========================================="

# 激活conda环境
source ~/.bashrc
conda activate unsloth

# 运行调试脚本
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/train
python debug_inference.py

echo ""
echo "=========================================="
echo "Debug completed!"
echo "End Time: $(date)"
echo "=========================================="