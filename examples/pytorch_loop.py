"""A plain PyTorch training loop (the LM3-BiRefNet / UNet style): per-epoch train + val metrics
and a grid of input / ground truth / prediction images.

    python examples/pytorch_loop.py
"""
from pathlib import Path

import torch
import torch.nn as nn

import runraccoon as wandb

# Your project's output dir; runs go to <OUTPUT_DIR>/runraccoon/run-<time>-<id>/
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

torch.manual_seed(0)
EPOCHS, BATCH = 15, 16

# toy segmentation: predict a mask of bright pixels
def batch():
    x = torch.rand(BATCH, 1, 32, 32)
    return x, (x > 0.6).float()

model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

wandb.init(project="raccoon-examples", name="pytorch-loop", dir=OUTPUT_DIR,
           config={"epochs": EPOCHS, "batch": BATCH, "lr": 1e-2})

for epoch in range(EPOCHS):
    model.train()
    tl = 0.0
    for _ in range(20):
        x, y = batch()
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tl += loss.item() / 20

    model.eval()
    with torch.no_grad():
        x, y = batch()
        logits = model(x)
        pred = (logits.sigmoid() > 0.5).float()
        iou = ((pred * y).sum() / ((pred + y).clamp(max=1).sum() + 1e-6)).item()
        vl = loss_fn(logits, y).item()

    wandb.log({"train/loss": tl, "val/loss": vl, "val/mIoU@0.5": iou, "lr": opt.param_groups[0]["lr"],
               "epoch": epoch})
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        # tensors are CHW; RunRaccoon handles them like wandb.Image does
        grid = torch.cat([x[0], y[0], logits[0].sigmoid()], dim=2)
        wandb.log({"val/inputs_gt_pred": wandb.Image(grid, caption="input | ground truth | prediction")},
                  commit=False)

wandb.finish()
