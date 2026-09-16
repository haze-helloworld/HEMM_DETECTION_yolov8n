

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import config


def count_files(folder: Path, extensions: set | None = None) -> int:
    """
    Count how many files exist directly inside `folder` (recursively),
    optionally restricted to a set of extensions.

    Parameters
    ----------
    folder : Path
        The directory to search. If it doesn't exist, we return 0
        instead of raising an error (a missing folder just means "0
        files", which is usually what you want when doing a health check).
    extensions : set[str] | None
        e.g. {".jpg", ".png"}. If None, every file is counted regardless
        of extension.

    Returns
    -------
    int : the number of matching files.

    Notes on the implementation
    ----------------------------
    `folder.rglob("*")` recursively lists every file AND sub-folder
    under `folder` ("r" = recursive glob, "*" = match everything).
    We then filter to only real files (f.is_file(), which excludes
    sub-directories themselves) and, optionally, to matching extensions.

    `sum(condition for x in iterable)` is a common Python idiom for
    "count how many items satisfy this condition" - True/False count as
    1/0 when summed.
    """
    if not folder.exists():
        return 0

    files = list(folder.rglob("*"))

    if extensions:
        return sum(
            f.is_file() and f.suffix.lower() in extensions
            for f in files
        )
    return sum(f.is_file() for f in files)


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse ONE YOLO-format label .txt file into a list of tuples:
        (class_id, x_center, y_center, width, height)

    If the file doesn't exist, we return an empty list rather than
    raising an error - a missing label file is treated the same as an
    image with zero objects (a valid "background" image).

    Lines that don't have exactly 5 whitespace-separated values are
    silently skipped. This is a lightweight sanity filter; for a more
    thorough audit that actually REPORTS malformed lines instead of
    hiding them, see `dataset_statistics()` below.
    """
    labels = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = map(float, parts)
        labels.append((int(cls), xc, yc, w, h))

    return labels


def yolo_box_to_pixels(
    xc: float, yc: float, w: float, h: float, img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert a YOLO-format normalized, center-based box into pixel-space
    corner coordinates (x1, y1, x2, y2) - i.e. top-left and bottom-right
    corners, in actual pixels, suitable for drawing with OpenCV/matplotlib.

    This is the exact conversion the notebook's ground-truth visualization
    cell performed inline; pulling it into a function means we (and any
    other script) can reuse it without copy-pasting the math.

    Example
    -------
    A box with xc=0.5, yc=0.5, w=0.2, h=0.4 on a 640x480 image is centered
    in the middle of the frame, 128px wide and 192px tall.
    """
    x1 = int((xc - w / 2) * img_width)
    y1 = int((yc - h / 2) * img_height)
    x2 = int((xc + w / 2) * img_width)
    y2 = int((yc + h / 2) * img_height)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Dataset-level checks
# ---------------------------------------------------------------------------

def verify_dataset_structure(dataset_dir: Path = config.DATASET_DIR) -> None:
    """
    Confirm that all six expected sub-folders exist:
        images/{train,val,test}
        labels/{train,val,test}

    Raises FileNotFoundError immediately if anything is missing.

    WHY RAISE INSTEAD OF JUST WARNING?
    This is a deliberate "fail fast" pattern. If we let training start on
    an incomplete dataset, Ultralytics might error out confusingly deep
    inside its own code, or - worse - silently train on a partial split
    without telling you. Stopping here, with a clear message, is much
    easier to debug.
    """
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "images" / "test",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "labels" / "test",
    ]

    print("Checking dataset structure...")
    for d in required_dirs:
        status = "OK " if d.exists() else "MISSING"
        print(f"  [{status}] {d}")

    if not all(d.exists() for d in required_dirs):
        raise FileNotFoundError(
            "One or more required train/val/test directories are missing. "
            "Make sure your dataset is laid out as:\n"
            "  ThroughTheFog_full/images/{train,val,test}\n"
            "  ThroughTheFog_full/labels/{train,val,test}"
        )


def count_split_files(dataset_dir: Path = config.DATASET_DIR) -> Dict[str, Dict[str, int]]:
    """
    Count images and label files in each of train/val/test.

    Returns a nested dict like:
        {
          "train": {"images": 8901, "labels": 8901},
          "val":   {"images": 1780, "labels": 1780},
          "test":  {"images": 2019, "labels": 2019},
        }

    A large image/label count MISMATCH within a split usually signals a
    bug in how the dataset was split or exported (some legitimate
    mismatch is fine - background-only images may have no label file).
    """
    split_counts: Dict[str, Dict[str, int]] = {}

    for split in ["train", "val", "test"]:
        n_images = count_files(dataset_dir / "images" / split, config.IMAGE_EXTS)
        n_labels = count_files(dataset_dir / "labels" / split, {".txt"})
        split_counts[split] = {"images": n_images, "labels": n_labels}
        print(f"{split:5s} | images: {n_images:6d} | labels: {n_labels:6d}")

    return split_counts


def dataset_statistics(
    split: str, dataset_dir: Path = config.DATASET_DIR
) -> Tuple[Counter, List[tuple], List[tuple]]:
    """
    Thoroughly audit every label file in one split ("train"/"val"/"test").

    Returns a 3-tuple:
        counter      - Counter mapping class_id -> number of objects of
                        that class (only counts VALID rows)
        malformed    - list of (file_path, line_number, raw_line) for
                        rows that didn't have exactly 5 fields, or where
                        the fields weren't valid numbers
        invalid_ids  - list of (file_path, line_number, class_id) for
                        rows with a class ID outside the valid range
                        (e.g. a stray "3" when you only declared 3
                        classes numbered 0-2)

    WHY BOTHER WITH THIS LEVEL OF DETAIL?
    A single bad label line usually won't crash training - Ultralytics'
    data loader is fairly permissive - but it CAN silently corrupt your
    metrics (an out-of-range class ID might get dropped, or worse,
    misinterpreted). Catching these here, with the exact file and line
    number, makes them trivial to go fix by hand.
    """
    label_dir = dataset_dir / "labels" / split
    counter: Counter = Counter()
    malformed: List[tuple] = []
    invalid_ids: List[tuple] = []

    for label_path in label_dir.rglob("*.txt"):
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.strip().split()

            if not parts:
                # Blank line - harmless, just skip it.
                continue

            if len(parts) != 5:
                malformed.append((str(label_path), line_no, line))
                continue

            try:
                cls, xc, yc, w, h = map(float, parts)
            except ValueError:
                # One of the 5 fields wasn't a valid number.
                malformed.append((str(label_path), line_no, line))
                continue

            if int(cls) not in range(config.NUM_CLASSES):
                invalid_ids.append((str(label_path), line_no, int(cls)))
                continue

            counter[int(cls)] += 1

    return counter, malformed, invalid_ids


def print_dataset_statistics(splits: List[str] = ("train", "val")) -> Dict[str, Counter]:
    """
    Convenience wrapper: run `dataset_statistics()` for each split in
    `splits` and pretty-print a per-class object count summary.

    We deliberately default to only train/val (NOT test) - it's good
    practice to avoid even casually inspecting test-set statistics until
    the very end of a project, so you're not tempted to make decisions
    based on it (see the "train/val/test discipline" note in the
    accompanying PDF guide, Part 1, Section 1.8).
    """
    stats: Dict[str, Counter] = {}

    for split in splits:
        counter, malformed, invalid = dataset_statistics(split)
        stats[split] = counter

        print(f"\n{split.upper()}")
        for cls_id, name in enumerate(config.CLASS_NAMES):
            print(f"  {name:10s}: {counter.get(cls_id, 0)} objects")
        print("  malformed lines :", len(malformed))
        print("  invalid class IDs:", len(invalid))

    return stats


def write_data_yaml(
    dataset_dir: Path = config.DATASET_DIR,
    data_yaml_path: Path = config.DATA_YAML,
    class_names: List[str] = config.CLASS_NAMES,
) -> None:
    """
    Write (or overwrite) the data.yaml file Ultralytics needs to find the
    dataset. Generating this file FROM CODE, rather than hand-editing a
    YAML file, guarantees the class names/order always match
    `config.CLASS_NAMES` used everywhere else in this project - one less
    place for a human typo to cause a class-ID mismatch between your
    labels and your model's predictions.

    `dataset_dir.as_posix()` converts the path to forward-slash form
    (even on Windows), which YAML/Ultralytics parses more reliably than
    backslash-separated Windows paths.
    """
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))

    yaml_text = (
        f'path: "{dataset_dir.as_posix()}"\n'
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n"
    )

    data_yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {data_yaml_path}:\n")
    print(data_yaml_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    verify_dataset_structure()
    print()
    count_split_files()
    write_data_yaml()
    print_dataset_statistics()
