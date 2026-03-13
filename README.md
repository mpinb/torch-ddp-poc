# DDP "Inconsistent 0 Params" — CUDA Driver Bug PoC

A minimal proof-of-concept for reproducing and diagnosing the following
PyTorch Distributed Data Parallel (DDP) error, which has been observed on
compute nodes with a faulty or stale CUDA driver state:

```
RuntimeError: DDP expects same model across all ranks, but Rank 0 has 34 params,
while rank 1 has inconsistent 0 params.
```

This repository is intended to be used as a diagnostic tool to test compute
nodes **before and after** a CUDA driver update, and to share a reproducible
case with colleagues investigating the same class of failure.

---

## Repository Contents

| File | Description |
|---|---|
| `ddp_poc.py` | Python PoC script — two modes of reproduction (see below) |
| `ddp_poc.sbatch` | SLURM batch script — pins the job to a specific node and captures diagnostics |
| `README.md` | This file |

---

## Background: Root Cause

### The bug

When `CUDA_MODULE_LOADING=LAZY` is active, CUDA kernels and the CUDA context
itself are loaded on first use rather than at process startup. This setting has
been the **default since NVIDIA driver ≥ 510** and is generally benign.

However, on a node whose GPU is in a stale or partially-reset driver state
(e.g. from a job that previously crashed without releasing the GPU cleanly),
the lazy loader **silently fails** to materialise a valid CUDA context on
one or more ranks. The rank does not raise an exception at this point — it
simply has no usable GPU context.

When DDP initialises, it broadcasts the parameter count from every rank as a
consistency check. The rank with the failed CUDA context reports **0 parameters**
(it never built the model), while the healthy rank reports the correct count.
DDP then raises the `RuntimeError` above.

### Why the job hangs instead of terminating cleanly

After the `RuntimeError`, NCCL's collective watchdog is still running with a
long timeout (often 10–30 minutes). The job does not terminate until that
timeout fires or the user cancels it manually. Cancelling a hung NCCL job
causes SLURM to drain the node.

### Upstream references

- [pytorch/pytorch #102438](https://github.com/pytorch/pytorch/issues/102438) — "DDP multi node multi gpu inconsistent params"
- [huggingface/accelerate #1481](https://github.com/huggingface/accelerate/issues/1481) — related report with identical error string

---

## Why 34 Parameters?

The `TinyNet` model in `ddp_poc.py` was deliberately constructed to produce
exactly **34 learnable parameters**, matching the count in the original error
message reported by the cluster user. This makes the PoC output directly
comparable to the failure observed in production.

| Layer | Formula | Count |
|---|---|---|
| `Linear(4 → 4)` | 4×4 weights + 4 bias | 20 |
| `BatchNorm1d(4)` | 4 weight + 4 bias | 8 |
| `Linear(4 → 2)` | 4×2 weights + 2 bias | 6 |
| **Total** | | **34** |

---

## Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.12 with `torch.distributed` support
- For **Mode A** (hardware): a node with ≥ 2 GPUs and NCCL installed
- For **Mode B** (software simulation): any machine, no GPU required

Install dependencies if needed:

```bash
pip install torch
```

---

## Running the PoC

### Mode A — Hardware test on a real node

This mode runs the PoC against actual GPUs. On a **healthy** node the script
completes successfully and prints a confirmation for each rank. On a node with
a faulty CUDA driver state (such as `somagpu095`) the `RuntimeError` is raised
within seconds of DDP initialisation.

**Directly with `torchrun`** (requires ≥ 2 GPUs on the current machine):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py
```

**Via SLURM** (pins the job to the target node):

```bash
# Edit --nodelist in ddp_poc.sbatch to point at the node under test, then:
sbatch ddp_poc.sbatch
```

Output logs are written to `ddp_poc_<jobid>.out` and `ddp_poc_<jobid>.err`.

**Expected output on a healthy node:**

```
[rank 0/2] host=somagpu095  device=cuda:0  CUDA_MODULE_LOADING=LAZY  simulate_bug=False
[rank 1/2] host=somagpu095  device=cuda:1  CUDA_MODULE_LOADING=LAZY  simulate_bug=False
[rank 0] local param count before DDP: 34
[rank 1] local param count before DDP: 34
[rank 0] DDP init succeeded — 34 params
[rank 1] DDP init succeeded — 34 params
[rank 0] training step OK — loss=0.7312
[rank 1] training step OK — loss=0.7312
[rank 0] done.
[rank 1] done.
```

**Expected output on a faulty node:**

```
[rank 0/2] host=somagpu095  device=cuda:0  CUDA_MODULE_LOADING=LAZY  simulate_bug=False
[rank 1/2] host=somagpu095  device=cuda:1  CUDA_MODULE_LOADING=LAZY  simulate_bug=False
[rank 0] local param count before DDP: 34
[rank 1] local param count before DDP: 0
======================================================================
[rank 0] *** BUG REPRODUCED ***
[rank 0] RuntimeError: DDP expects same model across all ranks, but Rank 0
has 34 params, while rank 1 has inconsistent 0 params.
======================================================================
```

---

### Mode B — Software simulation (no GPU required)

This mode makes rank 1 deliberately skip model construction, reproducing the
exact 0-param scenario without needing a faulty GPU. It is useful for:

- Verifying the error message and PoC script on any machine before deploying to the cluster
- Demonstrating the bug to colleagues who do not have direct cluster access

```bash
SIMULATE_BUG=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py
```

You can also trigger Mode B through SLURM by editing `ddp_poc.sbatch` and
setting `SIMULATE_BUG=1`, then submitting as normal.

**Expected output:**

```
[rank 0/2] host=...  device=cpu  simulate_bug=True
[rank 1/2] host=...  device=cpu  simulate_bug=True
[rank 0] local param count before DDP: 34
[rank 1] BUG SIMULATION: skipping model construction (pretending CUDA context init failed silently)
[rank 1] local param count before DDP: 0
======================================================================
[rank 0] *** BUG REPRODUCED ***
[rank 0] RuntimeError: DDP expects same model across all ranks, but Rank 0
has 34 params, while rank 1 has inconsistent 0 params.
======================================================================
```

---

## Recommended Diagnostic Steps

Use the following sequence when testing a suspect node. Steps 1–2 can be done
before any CUDA driver update; repeat Steps 3–4 after the update to confirm
the fix.

### Step 1 — Verify the error is reproducible in software (Mode B)

Run Mode B on any machine to confirm the PoC is working correctly in your
environment and that the error message matches what was observed in production.

```bash
SIMULATE_BUG=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py
```

### Step 2 — Run the hardware test on the suspect node (Mode A)

Submit the SLURM job pinned to the faulty node. The `.sbatch` script
automatically runs `nvidia-smi` to capture GPU health and ECC error counts
**before** the Python job starts.

```bash
# Confirm --nodelist=somagpu095 (or the node under test) in ddp_poc.sbatch
sbatch ddp_poc.sbatch
```

Inspect the SLURM output files:

```bash
# Check for the RuntimeError and NCCL warnings
cat ddp_poc_<jobid>.err

# Check the nvidia-smi snapshot and rank-level output
cat ddp_poc_<jobid>.out
```

Look for the following indicators of a driver-level fault in the `.err` file:

- `NCCL WARN` or `cudaErrorUnknown` lines near DDP initialisation
- ECC uncorrected volatile error counts > 0 in the `nvidia-smi` table
- Any rank reporting `device=cpu` when GPUs are expected

### Step 3 — Apply the temporary mitigation (optional, pre-update)

While waiting for a full driver update, add the following to the SLURM job
prologue or to users' job scripts for the affected node. This forces immediate
CUDA context creation, which either fails loudly at startup (exposing the fault
early) or prevents the silent lazy-load failure entirely.

```bash
export CUDA_MODULE_LOADING=EAGER
```

### Step 4 — Retest after the CUDA driver update

After updating the driver on the node, re-run Mode A **without** the `EAGER`
override and confirm:

1. The script completes without the `RuntimeError`.
2. Both ranks report 34 params and a successful training step.
3. `nvidia-smi` shows 0 uncorrected ECC errors.
4. No `NCCL WARN` lines appear in the `.err` file.

A clean run of Mode A is the acceptance criterion for the node being returned
to service.

---

## Quick Reference

| Scenario | Command |
|---|---|
| Software simulation (any machine) | `SIMULATE_BUG=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py` |
| Hardware test via `torchrun` | `torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_poc.py` |
| Hardware test via SLURM | `sbatch ddp_poc.sbatch` |
| Temporary mitigation | `export CUDA_MODULE_LOADING=EAGER` in job script |
