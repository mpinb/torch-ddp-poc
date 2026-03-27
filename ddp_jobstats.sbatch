#!/bin/bash
#SBATCH --job-name=ddp_jobstats
#SBATCH --partition=GPU-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/ddp_jobstats_%j.out
#SBATCH --error=logs/ddp_jobstats_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

export PYTHONFAULTHANDLER=1

echo "========================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "========================================================"

# Query raw driver / GPU health before launching the Python job
nvidia-smi --query-gpu=index,name,driver_version,pstate,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
echo "--------------------------------------------------------"

# Source the Python environment (adjust as needed)
source ~/anaconda3/bin/activate torch-ddp
echo "Activated Python environment: $(which python)"

srun python ddp_jobstats.py
