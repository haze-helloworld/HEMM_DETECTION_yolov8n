

import random
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from src.dataset_utils import load_yolo_labels, yolo_box_to_pixels


def show_ground_truth_samples(
    split: str = "train",
    n: int = 9,
    seed: int = 42,
    dataset_dir: Path = config.DATASET_DIR,
) -> None:
  

    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in config.IMAGE_EXTS]
    if not images:
        print(f"No images found in {image_dir}")
        return

    random.seed(seed)
    selected = random.sample(images, min(n, len(images)))

    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    # .reshape(-1) flattens the axes grid into a 1D list, regardless of
    # whether matplotlib gave us a 2D array (multiple rows), a 1D array
    # (one row), or a single Axes object (one image total).
    axes = np.array(axes).reshape(-1)

    for ax, image_path in zip(axes, selected):
        # cv2.imread loads in BGR channel order (an OpenCV historical
        # quirk) - we must convert to RGB before handing the array to
        # matplotlib, or colors will look wrong (e.g. blue/red swapped).
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"

        for cls, xc, yc, bw, bh in load_yolo_labels(label_path):
            x1, y1, x2, y2 = yolo_box_to_pixels(xc, yc, bw, bh, w, h)

            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
            )
            class_label = (
                config.CLASS_NAMES[cls] if cls < len(config.CLASS_NAMES) else f"class_{cls}"
            )
            ax.text(x1, max(0, y1 - 5), class_label, fontsize=10)

        ax.imshow(img)
        ax.set_title(image_path.name, fontsize=9)
        ax.axis("off")

    # If we asked for 9 images but only 7 existed, turn off the unused
    # subplot axes so they don't render as empty white boxes.
    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_class_distribution(counts_by_class: List[int], title: str = "Object Distribution") -> None:
  
    plt.figure(figsize=(8, 5))
    bars = plt.bar(config.CLASS_NAMES, counts_by_class)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of annotated objects")

    # Label each bar with its exact count, formatted with thousands
    # separators (e.g. "4,210") for readability.
    for bar, value in zip(bars, counts_by_class):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def show_prediction_grid(results, image_paths: List[Path], cols: int = 3) -> None:
   
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, result, path in zip(axes, results, image_paths):
        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        ax.set_title(Path(path).name, fontsize=9)
        ax.axis("off")

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
