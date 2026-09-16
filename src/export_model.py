"""
src/export_model.py
=====================
Convert the trained PyTorch checkpoint (best.pt) into ONNX format for
deployment. This is Section 17 of the original notebook.

WHAT IS ONNX AND WHY EXPORT TO IT?
-------------------------------------
ONNX (Open Neural Network Exchange, https://onnx.ai/) is an open,
framework-independent file format for trained models. A `.pt` file only
runs inside PyTorch. A `.onnx` file can run inside many different
"runtimes" (onnxruntime, TensorRT, OpenVINO, etc.), on many different
languages/platforms (Python, C++, C#, web via ONNX Runtime Web), without
needing a full PyTorch install on the machine that runs inference.

For this project: the CENTRAL WORKSTATION that will run inference on
live CCTV feeds doesn't need to be the same machine (or even the same
OS/Python setup) that trained the model. Exporting to ONNX decouples
"where we trained" from "where we deploy".

IMPORTANT: keep best.pt even after exporting. ONNX export can
occasionally introduce small numerical differences, or not perfectly
support every custom post-processing step - best.pt remains the
authoritative "source of truth" model, and re-exporting from it is
always possible if something looks off in the ONNX version.

HOW TO RUN
-----------
    python -m src.export_model
    python -m src.export_model --half        # export at fp16 precision (smaller, faster)
"""

import argparse

from ultralytics import YOLO

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export best.pt to ONNX format.")
    parser.add_argument(
        "--weights", type=str, default=str(config.BEST_MODEL_PATH),
        help="Path to the trained .pt checkpoint to export.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMAGE_SIZE,
        help="Fixed input size to bake into the exported ONNX graph.",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Export at FP16 (half) precision instead of FP32. "
             "Roughly halves file size and can speed up inference on "
             "hardware/runtimes that support fp16, at a small (usually "
             "negligible) accuracy cost worth checking on your val set.",
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="Run the ONNX graph simplifier (removes redundant nodes). "
             "Generally safe to leave on.",
    )
    return parser.parse_args()


def export_to_onnx(args: argparse.Namespace) -> str:
    model = YOLO(args.weights)

    # model.export(...) does the actual conversion. Ultralytics traces
    # the PyTorch computation graph and serializes it into ONNX's
    # standard operator set, embedding the fixed input size (imgsz) into
    # the exported graph - the exported model will always expect
    # images of exactly this size.
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        simplify=args.simplify,
    )

    print(f"\nExported ONNX model to: {exported_path}")
    print("Keep the original .pt checkpoint too - it remains the source of truth.")
    return exported_path


if __name__ == "__main__":
    export_to_onnx(parse_args())
