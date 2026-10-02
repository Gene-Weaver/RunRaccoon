"""Ultralytics YOLO with RunRaccoon instead of wandb.

Ultralytics has a built-in wandb callback that does `import wandb`. Because importing
RunRaccoon registers it as `wandb`, that callback logs locally - losses, mAP, PR/F1 curves,
confusion matrices, train/val batch images - with no other changes.

    python examples/ultralytics_yolo.py --data coco8.yaml --model yolo11n.pt --epochs 10
"""
import argparse

import runraccoon as wandb              # must come before training starts (it replaces `import wandb`)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="coco8.yaml")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="0")
ap.add_argument("--project", default="training_progress")
ap.add_argument("--name", default="yolo_example")
args = ap.parse_args()

from ultralytics import YOLO, settings  # noqa: E402

settings.update({"wandb": True})        # turns on Ultralytics' wandb callback (now RunRaccoon)

# Optional: start the run yourself to choose its name/config. Otherwise Ultralytics starts one.
wandb.init(project="raccoon-yolo", name=args.name, config=vars(args), dir=f"{args.project}/{args.name}")
run_id = wandb.run.id


def per_epoch_qc(trainer):
    """Your own per-epoch logging goes to the same run and step as Ultralytics' metrics."""
    wandb.log({"qc/fitness": float(trainer.fitness or 0)}, step=trainer.epoch + 1)


model = YOLO(args.model)
model.add_callback("on_fit_epoch_end", per_epoch_qc)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=args.device,
            project=args.project, name=args.name, exist_ok=True)

# Ultralytics finishes the run when training ends; reopen it to attach held-out test results.
test = YOLO(f"{args.project}/{args.name}/weights/best.pt").val(data=args.data, split="val", device=args.device)
if wandb.run is None:
    wandb.init(project="raccoon-yolo", id=run_id, resume="must", dir=f"{args.project}/{args.name}")
wandb.summary.update({f"test/{k}": float(v) for k, v in test.results_dict.items()})
wandb.finish()
