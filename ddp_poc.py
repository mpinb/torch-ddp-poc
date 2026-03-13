"""
Proof-of-Concept: DDP "inconsistent 0 params" bug
===================================================
Linked upstream reports
  - pytorch/pytorch #102438
  - huggingface/accelerate #1481

Root cause
----------
When CUDA_MODULE_LOADING=LAZY is set (default for NVIDIA driver ≥ 510),
CUDA kernels are loaded on first use rather than at context creation.
If a GPU on a compute node has a stale / partially-reset driver state,
the lazy loader silently fails to materialise the CUDA context on that
rank.  DDP then broadcasts parameter counts across ranks and finds
rank N reporting 0 params, raising:

    RuntimeError: DDP expects same model across all ranks, but
    Rank 0 has 34 params, while rank 1 has inconsistent 0 params.

The job then hangs (NCCL watchdog fires long after the SLURM wall-time
if the user does not cancel manually), and the node drains.

Reproducing the bug (mode A — hardware fault simulation)
---------------------------------------------------------
Run with CUDA_MODULE_LOADING=LAZY *and* CUDA_VISIBLE_DEVICES pointing
rank 1 at a GPU whose driver is unhealthy.  On a healthy cluster this
script should succeed; on somagpu095 with the known driver fault it
should raise the error within seconds of DDP init.

Reproducing the bug (mode B — software simulation, no GPU needed)
------------------------------------------------------------------
Set the environment variable SIMULATE_BUG=1 before launching.
Rank 1 will deliberately skip model construction so that DDP sees
0 params there, reproducing the exact error message.

Usage
-----
# Software simulation (any machine):
SIMULATE_BUG=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py

# Hardware test on somagpu095 (requires ≥2 GPUs):
torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py

# Via SLURM (see ddp_poc.sbatch):
sbatch ddp_poc.sbatch
"""

import os
import sys
import socket
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ── tiny model identical to what triggers the 34-param count ──────────────────
class TinyNet(nn.Module):
    """
    34 learnable parameters
      Linear(4→4): 4*4 weights + 4 bias = 20
      Linear(4→2): 4*2 weights + 2 bias =  6
      BatchNorm1d(4): 4 weight + 4 bias  =  8
    Total = 34  ← matches the count in the user's error message
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)      # 20 params
        self.bn  = nn.BatchNorm1d(4)    # 8  params
        self.fc2 = nn.Linear(4, 2)      # 6  params

    def forward(self, x):
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    # ── distributed setup ────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    hostname   = socket.gethostname()

    simulate_bug = os.environ.get("SIMULATE_BUG", "0") == "1"

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    print(f"[rank {rank}/{world_size}] host={hostname}  device={device}  "
          f"CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING', 'not set')}  "
          f"simulate_bug={simulate_bug}",
          flush=True)

    # ── model construction ───────────────────────────────────────────────────
    if simulate_bug and rank != 0:
        # Mimic what happens when a faulty GPU silently skips CUDA init:
        # the rank ends up with no model at all, so DDP sees 0 params.
        print(f"[rank {rank}] BUG SIMULATION: skipping model construction "
              f"(pretending CUDA context init failed silently)", flush=True)
        model = nn.Module()          # empty — 0 params
    else:
        model = TinyNet().to(device)

    param_count = count_params(model)
    print(f"[rank {rank}] local param count before DDP: {param_count}", flush=True)

    # ── DDP wrapping (this is where the error surfaces) ──────────────────────
    try:
        if torch.cuda.is_available():
            ddp_model = DDP(model, device_ids=[device.index])
        else:
            ddp_model = DDP(model)

        print(f"[rank {rank}] DDP init succeeded — {count_params(ddp_model)} params",
              flush=True)

    except RuntimeError as exc:
        # Pretty-print the exact error the cluster user saw
        print(f"\n{'='*70}", flush=True)
        print(f"[rank {rank}] *** BUG REPRODUCED ***", flush=True)
        print(f"[rank {rank}] {exc}", flush=True)
        print(f"{'='*70}\n", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

    # ── minimal training step to confirm gradients flow ──────────────────────
    if not simulate_bug:
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)
        x = torch.randn(8, 4, device=device)
        y = torch.randint(0, 2, (8,), device=device)
        loss = nn.CrossEntropyLoss()(ddp_model(x), y)
        loss.backward()
        optimizer.step()
        print(f"[rank {rank}] training step OK — loss={loss.item():.4f}", flush=True)

    dist.destroy_process_group()
    print(f"[rank {rank}] done.", flush=True)


if __name__ == "__main__":
    main()
