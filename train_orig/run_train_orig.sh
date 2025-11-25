#!/bin/bash
#SBATCH --job-name=train_orig      # 作业名称
#SBATCH --output=logs/train_orig_%j.out   # 标准输出日志 (%j会被作业ID替换)
#SBATCH --error=logs/train_orig_%j.err    # 错误输出日志
#SBATCH --partition=project
#SBATCH --account=cs707
#SBATCH --qos=cs707qos
#SBATCH --gres=gpu:1                 # 请求1个GPU
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

# 激活conda环境（如果需要）
source ~/.bashrc  # 或 source ~/miniconda3/etc/profile.d/conda.sh
conda activate unsloth

# 打印GPU信息
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# 打印Python和PyTorch信息
echo "Python Version:"
python --version
echo ""

echo "PyTorch Version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
echo ""

# 检查unsloth是否安装
echo "Checking Unsloth installation:"
python -c "import unsloth; print(f'Unsloth version: {unsloth.__version__}')" || echo "Unsloth not installed. Installing..."
if [ $? -ne 0 ]; then
    echo "Installing Unsloth..."
    pip install unsloth
fi
echo ""

# 进入训练目录
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/train_orig

# 运行训练脚本
echo "=========================================="
echo "Starting Training (Original Labels)..."
echo "=========================================="

TRAIN_DATA_PATH="/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_train_orig.jsonl"
TEST_DATA_PATH="/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test_orig.jsonl"
OUTPUT_DIR="./outputs"
export TRAIN_DATA_PATH TEST_DATA_PATH OUTPUT_DIR
 
python train_qwen3_unsloth_orig.py \
    --train_data $TRAIN_DATA_PATH \
    --test_data $TEST_DATA_PATH \
    --output_dir $OUTPUT_DIR

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with error code: $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi
