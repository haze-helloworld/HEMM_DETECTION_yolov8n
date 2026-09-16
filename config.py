"""
config.py
=========
Central configuration for the whole project.



HOW TO USE THIS FILE
---------------------
 Everything else in this
project (training, evaluation, inference, the API server) imports from
here instead of hard-coding paths.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT
# ---------------------------------------------------------------------------
# Path(__file__) is the path to THIS file (config.py).
# .resolve() turns it into an absolute path (no "..", no symlinks).
# .parent gives us the folder that contains this file, i.e. the project root.
#
# Using __file__ instead of Path.cwd() (like the original notebook did) is
# safer for a real project: Path.cwd() depends on *where you launched the
# script from*, which changes depending on how someone runs your code.
# __file__-based paths always resolve relative to the project itself.
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 2. DATASET PATHS
# ---------------------------------------------------------------------------
# The YOLO-format dataset used for TRAINING and EVALUATION.
# Expected layout (same as the original notebook):
#
#   ThroughTheFog_full/
#   ├── data.yaml
#   ├── images/{train,val,test}/
#   └── labels/{train,val,test}/
#
DATASET_DIR = PROJECT_ROOT / "ThroughTheFog_full"
DATA_YAML = DATASET_DIR / "data.yaml"

# Folder of single, unlabeled real-world images (e.g. exported CCTV frames
# or YouTube frames) used ONLY for one-off inference / pseudo-labeling,
# never for training directly.
ACTUAL_IMAGES_DIR = PROJECT_ROOT / "actual images"

# Where auto-generated ("pseudo") labels + annotated images get written
# after running infer_image.py on the folder above.
ACTUAL_OUTPUT_DIR = PROJECT_ROOT / "actual_images_labeled"
ACTUAL_OUTPUT_IMAGES = ACTUAL_OUTPUT_DIR / "images"
ACTUAL_OUTPUT_LABELS = ACTUAL_OUTPUT_DIR / "labels"

# Where trained model runs (weights, logs, plots) get saved.
# Ultralytics auto-increments this folder name (e.g. "yolov8n_fog_mine2")
# if you train more than once, so previous runs are never overwritten.
RUNS_DIR = PROJECT_ROOT / "runs"
TRAIN_RUN_NAME = "yolov8n_fog_mine_full"
VAL_RUN_NAME = "validation_evaluation"

# Convenience paths to the two most important training outputs.
TRAIN_RUN_DIR = RUNS_DIR / TRAIN_RUN_NAME
BEST_MODEL_PATH = TRAIN_RUN_DIR / "weights" / "best.pt"     # best checkpoint (by val metric)
LAST_MODEL_PATH = TRAIN_RUN_DIR / "weights" / "last.pt"      # checkpoint from the final epoch
RESULTS_CSV = TRAIN_RUN_DIR / "results.csv"                  # per-epoch metrics log

# The exported deployment-ready ONNX file lives next to best.pt after export.
ONNX_MODEL_PATH = TRAIN_RUN_DIR / "weights" / "best.onnx"

# ---------------------------------------------------------------------------
# 3. CLASS DEFINITIONS
# ---------------------------------------------------------------------------
# IMPORTANT: the order of this list defines the class IDs used everywhere.
# Index 0 = "human", index 1 = "vehicle", index 2 = "obstacle".
# This MUST match the "names:" section written into data.yaml, or your
# metrics/labels will be silently wrong (a model reporting "vehicle" when
# it means "human" is a serious bug for a safety system - always keep this
# list and data.yaml in sync).
CLASS_NAMES = ["human", "vehicle", "obstacle"]
NUM_CLASSES = len(CLASS_NAMES)

# File extensions we treat as "an image" when scanning folders.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# 4. TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
# These mirror the values used in the original notebook. Override them from
# the command line when running train.py (see that file's --help) instead
# of editing this file every time, if you want to experiment.
TRAIN_EPOCHS = 20      # how many full passes over the training set
IMAGE_SIZE = 640       # images are resized so their longest side is this many pixels
EARLY_STOP_PATIENCE = 10  # stop training if val performance hasn't improved for this many epochs
TRAIN_WORKERS = 2      # background CPU processes used to load/augment images

# ---------------------------------------------------------------------------
# 5. INFERENCE SETTINGS
# ---------------------------------------------------------------------------
# Confidence threshold = "how sure must the model be before we trust a
# detection?" Lower = more detections but more false alarms.
# Higher = fewer detections but fewer false alarms.
# We use two different values on purpose:
#   - a lower one for well-understood validation/test data (we trust the
#     ground truth, so we can afford to look at borderline detections too)
#   - a higher one for real-world / unverified footage, so a human
#     reviewer isn't swamped with noisy pseudo-labels.
VAL_CONFIDENCE = 0.25
DEPLOY_CONFIDENCE = 0.35

# NMS (Non-Maximum Suppression) IoU threshold: when the model predicts several
# overlapping boxes for the same real object, boxes with IoU above this value
# (against a higher-confidence box of the SAME class) are suppressed
# (removed) as duplicates. 0.45 is the Ultralytics default and a reasonable
# starting point; lower it if you see duplicate boxes on the same object,
# raise it if you see legitimately overlapping objects being merged into one.
NMS_IOU_THRESHOLD = 0.45


def ensure_directories() -> None:
    """
    Create every output directory this project writes to, if it doesn't
    already exist. Safe to call multiple times (mkdir(..., exist_ok=True)
    does nothing if the folder is already there).

    Call this once at the start of any script that writes files, so you
    never hit a "FileNotFoundError: no such directory" mid-run.
    """
    for directory in [
        RUNS_DIR,
        ACTUAL_OUTPUT_DIR,
        ACTUAL_OUTPUT_IMAGES,
        ACTUAL_OUTPUT_LABELS,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
