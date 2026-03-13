#!/bin/bash
#SBATCH --job-name=ddp_poc
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # one task per GPU
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/ddp_poc_%j.out
#SBATCH --error=logs/ddp_poc_%j.err
#SBATCH --nodelist=somagpu095,somagpu094        # pin to the A100 nodes
#SBATCH --qos=GPU-a100

# Create logs directory if it doesn't exist
mkdir -p logs

# ── optional: force the exact driver env that triggers the bug ────────────────
# NVIDIA driver ≥ 510 sets this automatically; uncomment to make it explicit.
# export CUDA_MODULE_LOADING=LAZY

# ── NCCL diagnostics (helps confirm driver / NVLink health) ──────────────────
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1

# ── toggle software simulation (set to 1 to reproduce without a broken GPU) ──
export SIMULATE_BUG=0   # change to 1 for software-only reproduction

echo "========================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "GPUs          : $CUDA_VISIBLE_DEVICES"
echo "SIMULATE_BUG  : $SIMULATE_BUG"
echo "CUDA_MODULE_LOADING: ${CUDA_MODULE_LOADING:-not set by SLURM}"
echo "========================================================"

# Query raw driver / GPU health before launching the Python job
nvidia-smi --query-gpu=index,name,driver_version,pstate,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
echo "--------------------------------------------------------"

# Source the Python environment (adjust as needed)
source ~/anaconda3/bin/activate torch-ddp
echo "Activated Python environment: $(which python)"

srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    ddp_poc.py
