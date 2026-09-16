"""
src/evaluate.py
=================
Everything related to MEASURING how good the trained model is. This
covers Sections 11-15 of the original notebook: training curves,
validation metrics, held-out test metrics, the confusion matrix, and
the confidence-threshold sweep.

KEY METRICS EXPLAINED (see the accompanying PDF guide for full detail)
-------------------------------------------------------------------------
Precision  - Of all predicted positive detections, how many were
             correct? High precision = few false alarms.
Recall     - Of all ground-truth objects, how many were detected?
             High recall = few missed objects.
mAP@0.50   - Mean Average Precision, counting a detection "correct" if
             its overlap (IoU) with a ground-truth box is >= 0.50.
             A relatively lenient localization requirement.
mAP@0.50:0.95 - The same idea, but averaged over 10 stricter IoU
             thresholds (0.50, 0.55, ..., 0.95). Rewards TIGHTLY
             localized boxes, not just roughly-right ones.

WHY EVALUATE ON *THREE* THINGS (train curves, val, test)?
-------------------------------------------------------------
- Training curves show whether the model is still learning, has
  plateaued, or is OVERFITTING (training loss keeps improving while
  validation loss stalls/worsens).
- The validation set is used throughout the project to make decisions
  (which checkpoint is "best", which confidence threshold to use).
- The test set is touched ONLY ONCE, right at the end, to report a
  final, unbiased number that hasn't been used to tune anything.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display  # only used if run inside a notebook/IPython
from ultralytics import YOLO

import config


def load_best_model(weights_path: Path = config.BEST_MODEL_PATH) -> YOLO:
    """
    Load the "best" checkpoint saved during training (selected by
    Ultralytics based on validation performance, NOT necessarily the
    weights from the very last epoch - early stopping or a late-training
    dip means the last epoch is not always the best one).

    Raises FileNotFoundError with a clear message if training hasn't
    been run yet, instead of letting a confusing error surface later.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Run `python -m src.train` first."
        )
    return YOLO(str(weights_path))


def show_training_curves(run_dir: Path = config.TRAIN_RUN_DIR) -> Optional[pd.DataFrame]:
    """
    Load results.csv (Ultralytics' per-epoch metrics log) and print the
    final few rows. Returns the DataFrame so a notebook/script can plot
    it further (see plot_metric_curves below).

    `.columns.str.strip()` removes accidental leading/trailing whitespace
    from column names - Ultralytics' CSV headers sometimes have a
    leading space (e.g. " train/box_loss"), which would otherwise
    silently break later lookups like `history["train/box_loss"]`.
    """
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"results.csv not found at {results_csv}")
        return None

    history = pd.read_csv(results_csv)
    history.columns = history.columns.str.strip()
    print(history.tail())
    return history


def plot_metric_curves(history: pd.DataFrame) -> None:
    """
    Plot each key training/validation metric individually (rather than
    relying only on Ultralytics' combined results.png) - larger, easier
    to read closely, and easier to paste one at a time into a report.

    The "(B)" suffix on metric names (e.g. metrics/precision(B)) refers
    to the Box-detection task head, as opposed to a Mask or Pose head in
    other YOLOv8 task variants - for a plain detector like this one,
    every metric is a "(B)" metric.
    """
    import matplotlib.pyplot as plt  # imported here to keep this optional for headless use

    plot_specs = [
        ("train/box_loss", "Training Box Loss"),
        ("train/cls_loss", "Training Classification Loss"),
        ("train/dfl_loss", "Training DFL Loss"),
        ("val/box_loss", "Validation Box Loss"),
        ("val/cls_loss", "Validation Classification Loss"),
        ("val/dfl_loss", "Validation DFL Loss"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP@0.50"),
        ("metrics/mAP50-95(B)", "mAP@0.50:0.95"),
    ]

    for column, title in plot_specs:
        if column in history.columns:
            plt.figure(figsize=(8, 4.5))
            plt.plot(history["epoch"], history[column], linewidth=2)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(column)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()


def evaluate_split(model: YOLO, split: str, conf: float = config.VAL_CONFIDENCE, plots: bool = True):
    """
    Run a full evaluation pass over one dataset split ("val" or "test")
    with the given model, and print the four headline metrics.

    Re-running evaluation "from scratch" here (rather than reusing
    whatever the last training-time validation pass computed) gives a
    clean, reproducible measurement, saved into its own runs/ sub-folder
    so it never gets mixed up with the training run's own logs.

    Returns the raw Ultralytics metrics object, in case you need
    per-class arrays (see per_class_metrics below) or anything else not
    surfaced by this convenience wrapper.
    """
    run_name = config.VAL_RUN_NAME if split == "val" else f"{split}_evaluation"

    metrics = model.val(
        data=str(config.DATA_YAML),
        split=split,
        imgsz=config.IMAGE_SIZE,
        conf=conf,
        device="cpu",         # evaluation is cheap; CPU keeps the GPU free for other work
        plots=plots,
        project=str(config.RUNS_DIR),
        name=run_name,
    )

    d = metrics.results_dict
    print(f"\n===== {split.upper()} SET RESULTS (conf={conf}) =====")
    for key in [
        "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    ]:
        if key in d:
            print(f"{key:30s}: {d[key]:.4f}")

    return metrics


def per_class_metrics(metrics) -> None:
    """
    Print PER-CLASS precision/recall/mAP arrays (one number per class,
    in config.CLASS_NAMES order: human, vehicle, obstacle).

    WHY THIS MATTERS: a single averaged mAP can hide a badly-performing
    class. For a safety system, checking "obstacle" separately from
    "human"/"vehicle" is essential - a high overall score can still mask
    poor performance on exactly the class you most need to trust.

    Wrapped in try/except because Ultralytics' internal attribute names
    can vary slightly across versions - we'd rather print a clear
    warning than crash the whole evaluation run over a missing attribute.
    """
    print("\n===== PER-CLASS RESULTS =====")
    try:
        box = metrics.box
        for i, name in enumerate(config.CLASS_NAMES):
            print(f"{name:10s} | mAP50: {box.ap50[i]:.4f} | mAP50-95: {box.ap[i]:.4f}")
    except Exception as e:
        print("Per-class arrays could not be read:", e)


def show_confusion_matrix(run_dir: Path) -> None:
    """
    Display the confusion matrix PNGs Ultralytics auto-saves whenever
    `plots=True` was passed to model.val().

    WHAT TO LOOK FOR: pay particular attention to the human<->obstacle
    and vehicle<->obstacle confusion cells (a person or vehicle wrongly
    called "obstacle", or vice-versa, is exactly the kind of error that
    could cause a safety system to under- or over-react), and to missed
    detections (false negatives) - the most dangerous failure mode for a
    safety-critical detector.
    """
    for name in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        path = run_dir / name
        if path.exists():
            print(name)
            img = Image.open(path)
            try:
                display(img)   # nice inline display if run inside Jupyter/IPython
            except NameError:
                img.show()      # fallback: open in the OS's default image viewer
        else:
            print(f"Not found: {path}")


def confidence_threshold_sweep(
    model: YOLO, thresholds=(0.10, 0.20, 0.25, 0.40, 0.50, 0.60)
) -> pd.DataFrame:
    """
    Evaluate the SAME trained weights repeatedly, only changing the
    confidence threshold each time, to build a small table showing how
    precision/recall/mAP trade off against each other.

    WHY THIS MATTERS FOR A SAFETY SYSTEM: there is no universally
    "correct" confidence threshold - it depends on how much you're
    willing to trade false alarms for missed detections. This sweep
    turns "what threshold should we deploy with?" into an explicit,
    documented decision instead of silently keeping whatever the
    library's default happens to be.
    """
    rows = []
    for conf in thresholds:
        m = model.val(
            data=str(config.DATA_YAML),
            split="val",
            imgsz=config.IMAGE_SIZE,
            conf=conf,
            device="cpu",
            plots=False,   # don't regenerate plots for every threshold - just want the numbers
            verbose=False,
        )
        d = m.results_dict
        rows.append({
            "confidence": conf,
            "precision": d.get("metrics/precision(B)", np.nan),
            "recall": d.get("metrics/recall(B)", np.nan),
            "mAP50": d.get("metrics/mAP50(B)", np.nan),
            "mAP50-95": d.get("metrics/mAP50-95(B)", np.nan),
        })

    df = pd.DataFrame(rows)
    print(df)
    return df


def build_final_summary_table(
    split_counts: dict, metrics_dict: dict, epochs: int, imgsz: int
) -> pd.DataFrame:
    """
    Collect everything computed elsewhere (dataset counts, headline
    metrics, hyperparameters) into one tidy two-column table, suitable
    for pasting directly into a report or slide deck.
    """
    import torch

    final_rows = {
        "Model": "YOLOv8n",
        "Dataset": "ThroughTheFog_full",
        "Train images": split_counts["train"]["images"],
        "Validation images": split_counts["val"]["images"],
        "Test images": split_counts["test"]["images"],
        "Classes": ", ".join(config.CLASS_NAMES),
        "Image size": imgsz,
        "Epochs": epochs,
        "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "Precision": metrics_dict.get("metrics/precision(B)", np.nan),
        "Recall": metrics_dict.get("metrics/recall(B)", np.nan),
        "mAP@0.50": metrics_dict.get("metrics/mAP50(B)", np.nan),
        "mAP@0.50:0.95": metrics_dict.get("metrics/mAP50-95(B)", np.nan),
    }
    table = pd.DataFrame(list(final_rows.items()), columns=["Metric", "Value"])
    print(table)
    return table


# ---------------------------------------------------------------------------
# Run this file directly for a full evaluation pass:
#     python -m src.evaluate
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.dataset_utils import count_split_files

    model = load_best_model()

    print("\n--- Training curves ---")
    history = show_training_curves()

    print("\n--- Validation set ---")
    val_metrics = evaluate_split(model, split="val")
    per_class_metrics(val_metrics)
    show_confusion_matrix(config.RUNS_DIR / config.VAL_RUN_NAME)

    print("\n--- Held-out test set (only run this ONCE you're done tuning) ---")
    test_metrics = evaluate_split(model, split="test")
    per_class_metrics(test_metrics)

    print("\n--- Confidence threshold sweep (validation set) ---")
    confidence_threshold_sweep(model)

    print("\n--- Final summary table ---")
    split_counts = count_split_files()
    build_final_summary_table(
        split_counts, val_metrics.results_dict, config.TRAIN_EPOCHS, config.IMAGE_SIZE
    )
