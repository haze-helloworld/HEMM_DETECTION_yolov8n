"""
src/infer_image.py
====================
Run the trained model over a FOLDER of individual, unlabeled real-world
images (e.g. exported CCTV frames or YouTube frames) and save both an
annotated (boxes-drawn) copy of each image and a YOLO-format label file
of the model's own predictions. This is Section 20 of the original
notebook ("Automatic Labeling of Real-World Images").

WHAT IS A "PSEUDO-LABEL"?
-----------------------------
A label generated automatically by the MODEL'S OWN predictions, as
opposed to a label a human verified. Pseudo-labels are useful as a
starting point for expanding your training set (a human reviewer
corrects them, then they get added to the dataset), but they are NOT
ground truth on their own.

DO NOT feed these labels straight into training without review -
a model trained on its own uncorrected mistakes will simply reinforce
those mistakes. Always have a human look at (and fix) these before
reusing them as training data.

HOW TO RUN
-----------
    python -m src.infer_image
    python -m src.infer_image --conf 0.5 --input "some/other/folder"
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference over a folder of real-world images.")
    parser.add_argument(
        "--weights", type=str, default=str(config.BEST_MODEL_PATH),
        help="Path to the trained model (.pt or .onnx).",
    )
    parser.add_argument(
        "--input", type=str, default=str(config.ACTUAL_IMAGES_DIR),
        help="Folder of unlabeled images to run inference on.",
    )
    parser.add_argument(
        "--output", type=str, default=str(config.ACTUAL_OUTPUT_DIR),
        help="Folder to save annotated images + pseudo-label .txt files into.",
    )
    parser.add_argument(
        "--conf", type=float, default=config.DEPLOY_CONFIDENCE,
        help="Confidence threshold. Higher than the validation default on "
             "purpose - this is UNVERIFIED footage, so a stricter bar "
             "reduces the number of noisy pseudo-labels a human reviewer "
             "later has to sift through.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help='GPU index (e.g. "0") or "cpu". Default: auto-detect.',
    )
    return parser.parse_args()


def run_batch_inference(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"{input_dir} does not exist. Create it and add some images, "
            f"or point --input at the correct folder."
        )

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS
    ]
    print(f"Found {len(image_paths)} images in {input_dir}")
    if len(image_paths) == 0:
        raise ValueError(f"No supported images found in {input_dir}.")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(args.weights)

    total_detections = 0
    images_with_detections = 0

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=config.IMAGE_SIZE,
            conf=args.conf,
            device=device,
            verbose=False,
        )
        result = results[0]

        # --- Save an annotated copy so a human can quickly eyeball quality ---
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(annotated_rgb).save(output_images_dir / image_path.name)

        # --- Write a YOLO-format pseudo-label .txt file ---
        label_path = output_labels_dir / f"{image_path.stem}.txt"
        detections_for_image = 0

        with open(label_path, "w", encoding="utf-8") as f:
            if result.boxes is not None:
                for box in result.boxes:
                    # .item() converts a single-element PyTorch tensor
                    # into a plain Python number.
                    cls_id = int(box.cls.item())

                    # box.xywhn[0] gives (x_center, y_center, width,
                    # height) already NORMALIZED to [0, 1] - i.e.
                    # already in the exact format YOLO label files use,
                    # so no manual conversion is needed here (contrast
                    # with visualize.py, where we convert FROM this
                    # format TO pixels, for drawing).
                    x, y, w, h = box.xywhn[0].tolist()

                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    detections_for_image += 1
                    total_detections += 1

        if detections_for_image > 0:
            images_with_detections += 1

    print("\n" + "=" * 60)
    print("BATCH INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Images processed       : {len(image_paths)}")
    print(f"Images with detections : {images_with_detections}")
    print(f"Total detections       : {total_detections}")
    print(f"Confidence threshold   : {args.conf}")
    print(f"Annotated images saved : {output_images_dir}")
    print(f"Pseudo-labels saved    : {output_labels_dir}")
    print("\nREMINDER: these are pseudo-labels, not verified ground truth.")
    print("Review and correct them by hand before adding to a training set.")


if __name__ == "__main__":
    run_batch_inference(parse_args())
