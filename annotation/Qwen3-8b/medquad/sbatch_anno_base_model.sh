#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=12          # Number of CPU to request for the job
#SBATCH --mem=64GB                  # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=1-00:00:00           # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=project                 # The partition you've been assigned
#SBATCH --account=cs707                     # The account you've been assigned
#SBATCH --qos=cs707qos                      # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=my.wang.2024@msc.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=anno_base_model          # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment to avoid conflicts
module purge

# Activate the unsloth conda environment (conda manages its own Python)
eval "$(conda shell.bash hook)"
conda activate unsloth

# Verify Python and torch are available
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"


# Change to annotation directory
cd /common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation/Qwen3-8b/medquad

# Submit your job to the cluster
# Pass all command-line arguments to the Python script
srun --gres=gpu:1 python anno_infer_base_model.py "$@"
