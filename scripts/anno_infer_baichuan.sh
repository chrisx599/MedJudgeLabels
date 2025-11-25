#!/bin/bash
#SBATCH --job-name=classify          # 作业名称
#SBATCH --output=/common/home/projectgrps/CS707/CS707G2/logs/classify_%j.out   # 标准输出日志 (%j会被作业ID替换)
#SBATCH --error=/common/home/projectgrps/CS707/CS707G2/logs/classify_%j.err    # 错误输出日志
#SBATCH --partition=project
#SBATCH --account=cs707
#SBATCH --qos=cs707qos
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --cpus-per-task=8            # CPU核心数
#SBATCH --mem=64G                    # 内存
#SBATCH --time=24:00:00              # 最长运行时间
#SBATCH --mail-user=sc.zhou.2025@phdcs.smu.edu.sg # Who should receive the email notifications

# 创建日志目录
mkdir -p logs

# 修改CUDA环境配置
module load CUDA/11.8.0
module load cuDNN/8.9.7.29-CUDA-11.8.0

# 禁用flash-attn安装,改为安装其他依赖
pip install wheel
pip install numpy==1.24.3
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.2
pip install accelerate==0.25.0
pip install sentencepiece
pip install protobuf
pip install einops
pip install flash_attn

# 激活conda环境
# source ~/.bashrc  # 或 source ~/miniconda3/etc/profile.d/conda.sh
# conda activate med  # 替换为你的环境名

# 打印环境信息
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

pwd
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/

# 运行你的脚本
python annotation/transformer_baichuan_infer.py \
    --input /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/medical_qa_data.jsonl \
    --output /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_baichuan.jsonl \
    --batch-size 1

echo "Job finished at: $(date)"

# cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/scripts
# sbatch anno_infer_baichuan.sh
