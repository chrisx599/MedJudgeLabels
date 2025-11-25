#!/bin/bash
#SBATCH --job-name=evaluate       # 作业名称
#SBATCH --output=logs/evaluate_%j.out   # 标准输出日志 (%j会被作业ID替换)
#SBATCH --error=logs/evaluate_%j.err    # 错误输出日志
#SBATCH --partition=project
#SBATCH --account=cs707
#SBATCH --qos=cs707qos
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8            # CPU核心数
#SBATCH --mem=64G                    # 内存
#SBATCH --time=24:00:00              # 最长运行时间

# 打印作业信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate unsloth

# 打印GPU信息
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# 进入训练目录
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/train

# 运行评估脚本
echo "=========================================="
echo "Starting Evaluation..."
echo "=========================================="

# 可以通过命令行参数自定义评估配置
# 例如: sbatch run_evaluate.sh --model_path ./outputs/lora_adapter --max_samples 1000

# --lora_adapter_path ./outputs/lora_adapter \

python evaluate.py \
    --predictions_file ./predictions_origin.json \
    --output ./evaluation_results_origin.json \
    --max_seq_length 2048

# 检查评估是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: ./evaluation_results.json"
    echo "Predictions saved to: ./predictions.json"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with error code: $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi