#!/bin/bash
#SBATCH --job-name=eval_fast      # 作业名称
#SBATCH --output=../logs/eval_fast_%j.out   # 标准输出日志
#SBATCH --error=../logs/eval_fast_%j.err    # 错误输出日志
#SBATCH --partition=project
#SBATCH --account=cs707
#SBATCH --qos=cs707qos
#SBATCH --cpus-per-task=2         # CPU核心数
#SBATCH --mem=8G                  # 内存（无需GPU）
#SBATCH --time=00:30:00           # 最长运行时间

# 打印作业信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 设置环境变量
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate unsloth

# 进入训练目录
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/train

# 运行快速评估（从已保存的预测结果）
echo "=========================================="
echo "Starting Fast Evaluation (from predictions)..."
echo "=========================================="

# 检查预测结果文件是否存在
if [ ! -f "./predictions.json" ]; then
    echo "错误: 预测结果文件 ./predictions.json 不存在"
    echo "请先运行完整评估: sbatch run_evaluate.sh"
    exit 1
fi

python evaluate.py \
    --from_predictions ./predictions.json \
    --output ./evaluation_results.json

# 检查评估是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Fast evaluation completed successfully!"
    echo "Results saved to: ./evaluation_results.json"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Fast evaluation failed with error code: $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi