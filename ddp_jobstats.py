"""
Minimal single-GPU training script to exercise the jobstats tool.

The model is deliberately wide and deep so the job keeps the GPU busy
for at least a few minutes, giving jobstats enough sampling intervals
to produce meaningful data.

# Via SLURM (see ddp_jobstats.sbatch):
sbatch ddp_jobstats.sbatch
"""

import time
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── tuneable knobs ────────────────────────────────────────────────────────────
INPUT_DIM   = 1024
HIDDEN_DIM  = 4096
NUM_LAYERS  = 8       # depth of the MLP trunk
NUM_CLASSES = 1000
BATCH_SIZE  = 512
NUM_STEPS   = 5_000
LOG_EVERY   = 500     # print a progress line every N steps


# ── model ─────────────────────────────────────────────────────────────────────
class BigNet(nn.Module):
    """Wide MLP designed to stress GPU memory and compute.

    Approximate parameter count with defaults:
      Input layer  : 1024 × 4096 + 4096          ≈   4.2 M
      7 hidden     : 7 × (4096 × 4096 + 4096×3)  ≈ 117.7 M  (Linear + BN)
      Output layer : 4096 × 1000 + 1000           ≈   4.1 M
      Total                                        ≈ 126 M params (~480 MB fp32)
    """
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU()]
        for _ in range(NUM_LAYERS - 1):
            layers += [
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
            ]
        layers.append(nn.Linear(HIDDEN_DIM, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print(f"device: {device}", flush=True)

    # ── model construction ───────────────────────────────────────────────────
    model = BigNet().to(device)
    print(f"parameter count: {count_params(model):,}", flush=True)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  total memory: {props.total_memory / 1e9:.1f} GB",
              flush=True)
        print(f"allocated after model load: "
              f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"step {step:>6}/{NUM_STEPS}  "
                f"loss={loss.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"elapsed={elapsed:.1f}s  "
                f"steps/s={step / elapsed:.1f}",
                flush=True,
            )

    print(f"done — total time {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()