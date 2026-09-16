"""
src/train.py
=============
Fine-tune a COCO-pretrained YOLOv8n model on our custom
human/vehicle/obstacle dataset. This is Section 9 of the original
notebook, turned into a runnable, configurable script.

WHAT IS "FINE-TUNING" / "TRANSFER LEARNING"?
-----------------------------------------------
Training a CNN from completely random weights to recognize objects well
usually needs millions of labeled images and huge compute - not
practical for a small custom dataset. Instead, we start from a model
that has ALREADY been trained on a big, general dataset (COCO: 80
everyday object classes) and continue training it, at a lower learning
rate, on our much smaller dataset. The model keeps its general
"how to see" knowledge (edges, shapes, textures) from COCO, and adapts
its final layers to our 3 classes.

`YOLO("yolov8n.pt")` loads those COCO-pretrained weights (downloading
them automatically the first time, if not already cached locally).

HOW TO RUN
-----------
    python -m src.train
    python -m src.train --epochs 50 --imgsz 640 --device 0
    python -m src.train --epochs 10 --device cpu     # force CPU

Run `python -m src.train --help` to see every option.
"""

import argparse

import torch
from ultralytics import YOLO

import config


def parse_args() -> argparse.Namespace:
    """
    Define the command-line arguments this script accepts.

    Using argparse (instead of hard-coded constants, like the original
    notebook cell) means you can experiment with different
    hyperparameters WITHOUT editing this file - e.g. to quickly try
    more epochs:

        python -m src.train --epochs 50
    """
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on the fog-mine dataset.")

    parser.add_argument(
        "--epochs", type=int, default=config.TRAIN_EPOCHS,
        help="Number of full passes over the training set.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMAGE_SIZE,
        help="Images are resized so their longest side is this many pixels.",
    )
    parser.add_argument(
        "--batch", type=int, default=-1,
        help="Batch size. -1 = auto-select the largest batch that fits in GPU memory.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help='GPU index as a string (e.g. "0") or "cpu". '
             "Default: auto-detect (GPU if available, else CPU).",
    )
    parser.add_argument(
        "--patience", type=int, default=config.EARLY_STOP_PATIENCE,
        help="Stop early if validation performance hasn't improved for this many epochs.",
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt",
        help="Starting weights. Default is COCO-pretrained YOLOv8-nano. "
             "Point this at an existing best.pt to continue fine-tuning further.",
    )
    parser.add_argument(
        "--run-name", type=str, default=config.TRAIN_RUN_NAME,
        help="Name of the run folder under runs/ where results are saved.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str | int:
    """
    Decide which device to train on.

    If the user explicitly passed --device, use that. Otherwise,
    auto-detect: use the first CUDA GPU (device 0) if PyTorch reports one
    is available, else fall back to CPU.

    Training a CNN on CPU is *possible* but can be 10-50x slower than on
    a GPU - this check matters a lot for how long training will take.
    """
    if device_arg is not None:
        # Ultralytics accepts either an int GPU index or the string "cpu".
        return int(device_arg) if device_arg.isdigit() else device_arg

    if torch.cuda.is_available():
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        return 0

    print("No CUDA GPU detected - training will run on CPU (this will be slow).")
    return "cpu"


def train(args: argparse.Namespace) -> None:
    """
    Run the actual fine-tuning job and print a short summary at the end.
    """
    device = resolve_device(args.device)

    # Load starting weights. If args.weights == "yolov8n.pt" and that
    # file isn't already cached locally, Ultralytics downloads the
    # official COCO-pretrained nano weights automatically.
    model = YOLO(args.weights)

    train_results = model.train(
        data=str(config.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=config.TRAIN_WORKERS,
        project=str(config.RUNS_DIR),
        name=args.run_name,
        pretrained=True,
        patience=args.patience,
        plots=True,     # auto-generate loss curves, PR curve, confusion matrix, etc.
        verbose=True,   # print detailed per-epoch progress
    )

    run_dir = config.RUNS_DIR / args.run_name
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Device used     : {device}")
    print(f"Run directory   : {run_dir}")
    print(f"Best checkpoint : {run_dir / 'weights' / 'best.pt'}")
    print(f"Results CSV     : {run_dir / 'results.csv'}")
    print("\nNext step: python -m src.evaluate")


if __name__ == "__main__":
    train(parse_args())
