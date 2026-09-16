"""
src/infer_stream.py
=====================
 multiple CCTV cameras all send their footage to ONE
CENTRAL WORKSTATION, which runs the AI model on every feed.


--------------------------
A `CameraWorker` class: one worker per camera feed. Each worker:
  1. Opens a video source (an RTSP URL, a local webcam, or a video file
     - all handled identically by OpenCV).
  2. Continuously reads frames from it.
  3. Runs YOLO inference on each frame.
  4. Stores the latest annotated frame + detections where other code
     (a dashboard, an API, an alert system) can read them.

And a `StreamManager` class that owns and coordinates several
`CameraWorker`s at once - one per camera - so a single Python process
running on the central workstation can watch every feed simultaneously.

WHY THREADS, NOT SEPARATE PROCESSES PER CAMERA?
----------------------------------------------------
Reading video frames from a network stream (RTSP) is mostly spent
WAITING for network I/O, not doing CPU work - a great fit for Python
threads, which can run I/O-bound work concurrently even though only one
thread executes Python bytecode at a time (the "GIL" - Global
Interpreter Lock). The actual YOLO inference call releases the GIL
while running on the GPU/native code, so multiple camera workers can
still make real progress in parallel on inference too. If you later
need true CPU-parallel work across many cameras with heavy CPU-side
pre/post-processing, look into Python's `multiprocessing` instead - but
for a handful of camera feeds, threads are simpler and usually enough.

HOW TO RUN A QUICK STANDALONE TEST
--------------------------------------
    python -m src.infer_stream --sources 0
        (source "0" = your laptop's built-in/first USB webcam - useful
         for a live demo without needing real CCTV hardware)

    python -m src.infer_stream --sources "rtsp://192.168.1.10/stream1" "rtsp://192.168.1.11/stream1"
        (two real camera feeds)

In production, you won't run this file directly - `app.py` imports
`StreamManager` and starts it as part of the backend service.
"""

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config


@dataclass
class Detection:
    """
    One detected object, in a plain, easy-to-serialize form (a
    dataclass, not a raw Ultralytics object) so it can be turned into
    JSON for the API layer or passed to a downstream module (e.g. a
    teammate's distance/time-to-collision code) without them needing to
    know anything about Ultralytics internals.
    """
    class_id: int
    class_name: str
    confidence: float
    # Bounding box in PIXEL coordinates (x1, y1) = top-left corner,
    # (x2, y2) = bottom-right corner, matching the resolution of the
    # frame this detection came from.
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class CameraState:
    """
    The latest known state of one camera feed - what any other part of
    the system (a web dashboard, an alert checker) should read to find
    out "what does this camera currently see?".

    Using a dataclass with sensible defaults means a freshly-created
    CameraState (before the first frame has even arrived) is still a
    valid, safe-to-read object.
    """
    camera_id: str
    connected: bool = False
    last_frame: Optional[np.ndarray] = None      # raw BGR frame (OpenCV format)
    last_annotated_frame: Optional[np.ndarray] = None  # frame with boxes drawn on it
    last_detections: List[Detection] = field(default_factory=list)
    last_update_time: float = 0.0
    fps: float = 0.0
    error: Optional[str] = None


class CameraWorker(threading.Thread):
    """
    Runs in its own background thread. Continuously reads frames from
    ONE camera source and updates a shared `CameraState` object with the
    latest frame, detections, and timing info.

    `threading.Thread` subclassing pattern: we override `run()`, which
    is what actually executes on the background thread once `.start()`
    is called. Everything before `.start()` (in `__init__`) runs
    normally on the calling thread.
    """

    def __init__(
        self,
        camera_id: str,
        source: str | int,
        model: YOLO,
        state: CameraState,
        conf: float = config.DEPLOY_CONFIDENCE,
        device: str | int = "cpu",
        target_fps: Optional[float] = None,
    ):
        # `daemon=True` means this background thread will not prevent
        # the whole Python process from exiting - important so Ctrl+C
        # or a normal shutdown doesn't hang waiting on camera threads.
        super().__init__(daemon=True)

        self.camera_id = camera_id
        self.source = source          # RTSP URL, video file path, or webcam index (0, 1, ...)
        self.model = model
        self.state = state
        self.conf = conf
        self.device = device
        # If set, we deliberately sleep between frames to avoid running
        # inference faster than needed (saves CPU/GPU if the downstream
        # system only needs, say, 5 detections/second per camera rather
        # than the camera's native 30fps).
        self.min_frame_interval = (1.0 / target_fps) if target_fps else 0.0

        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal this worker's run() loop to exit cleanly on its next iteration."""
        self._stop_event.set()

    def run(self) -> None:
        """
        The main loop for this camera. Runs until `.stop()` is called or
        the video source can't be read anymore.

        cv2.VideoCapture(source) handles RTSP URLs, local webcam indices,
        and video file paths identically - OpenCV figures out which kind
        of source it is from the string/int you give it.
        """
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.state.connected = False
            self.state.error = f"Could not open video source: {self.source}"
            print(f"[{self.camera_id}] ERROR: {self.state.error}")
            return

        self.state.connected = True
        print(f"[{self.camera_id}] Connected to {self.source}")

        last_frame_time = 0.0

        while not self._stop_event.is_set():
            # Throttle to target_fps if one was requested.
            if self.min_frame_interval:
                elapsed = time.time() - last_frame_time
                if elapsed < self.min_frame_interval:
                    time.sleep(self.min_frame_interval - elapsed)

            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                # Stream dropped/ended. For a live RTSP feed this often
                # means a network hiccup - in a production system you'd
                # add reconnect logic here. For clarity, we just log and
                # stop this worker's loop.
                self.state.connected = False
                self.state.error = "Stream ended or dropped (no more frames)."
                print(f"[{self.camera_id}] {self.state.error}")
                break

            detections = self._run_inference(frame)

            # Update the shared state. Because Python's GIL makes simple
            # attribute assignment effectively atomic, we don't need an
            # explicit lock here for this kind of "replace the whole
            # value" update - a reader will always see either the OLD
            # complete state or the NEW complete state, never a mix.
            self.state.last_frame = frame
            self.state.last_annotated_frame = self._draw_detections(frame, detections)
            self.state.last_detections = detections
            self.state.last_update_time = time.time()

            loop_duration = time.time() - loop_start
            self.state.fps = 1.0 / loop_duration if loop_duration > 0 else 0.0
            last_frame_time = time.time()

        cap.release()
        print(f"[{self.camera_id}] Worker stopped.")

    def _run_inference(self, frame: np.ndarray) -> List[Detection]:
        """
        Run the YOLO model on a single frame and convert Ultralytics'
        result object into our own simple `Detection` list.

        We pass the frame array directly (source=frame) rather than a
        file path - Ultralytics accepts NumPy arrays / OpenCV frames
        directly, which is exactly what we need for live video (there
        is no file to point to, only an in-memory frame).
        """
        results = self.model.predict(
            source=frame,
            imgsz=config.IMAGE_SIZE,
            conf=self.conf,
            iou=config.NMS_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        detections: List[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                # box.xyxy[0] gives PIXEL corner coordinates directly
                # (as opposed to box.xywhn used in infer_image.py, which
                # gives normalized center coordinates for saving to a
                # label file) - pixel corners are what we want here for
                # drawing on screen / sending to a downstream module
                # that reasons about actual frame coordinates.
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else f"class_{cls_id}",
                    confidence=confidence,
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                ))

        return detections

    @staticmethod
    def _draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw boxes + labels onto a COPY of the frame (we never mutate
        the original `frame`, since `state.last_frame` should stay the
        clean, undrawn-on image in case something downstream wants the
        raw pixels).
        """
        annotated = frame.copy()
        for det in detections:
            color = (0, 255, 0)  # green, BGR order (OpenCV convention)
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (det.x1, max(0, det.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return annotated


class StreamManager:
    """
    Owns and coordinates one `CameraWorker` per camera feed. This is the
    single object `app.py` (the backend server) talks to: "start
    watching these cameras", "give me the latest detections for camera
    X", "shut everything down".

    Think of this as the central workstation's "control room" - it
    doesn't do any inference itself, it just manages the workers that do.
    """

    def __init__(
        self,
        weights_path: str = str(config.BEST_MODEL_PATH),
        conf: float = config.DEPLOY_CONFIDENCE,
        device: str | int = "cpu",
    ):
        # Load the model ONCE and share it across all camera workers.
        # This is important: loading a YOLO model is relatively slow and
        # memory-hungry - you do NOT want to load a separate copy per
        # camera. Ultralytics' `.predict()` calls are safe to share
        # across threads for inference (they don't mutate persistent
        # model state between calls).
        self.model = YOLO(weights_path)
        self.conf = conf
        self.device = device

        self.workers: Dict[str, CameraWorker] = {}
        self.states: Dict[str, CameraState] = {}

    def add_camera(self, camera_id: str, source: str | int, target_fps: Optional[float] = None) -> None:
        """
        Start watching a new camera feed.

        `source` can be:
          - an RTSP URL string, e.g. "rtsp://192.168.1.10:554/stream1"
          - a local video file path, for testing with recorded footage
          - an integer (0, 1, ...) for a locally attached webcam
        """
        if camera_id in self.workers:
            raise ValueError(f"Camera '{camera_id}' is already registered.")

        state = CameraState(camera_id=camera_id)
        worker = CameraWorker(
            camera_id=camera_id,
            source=source,
            model=self.model,
            state=state,
            conf=self.conf,
            device=self.device,
            target_fps=target_fps,
        )

        self.states[camera_id] = state
        self.workers[camera_id] = worker
        worker.start()

    def remove_camera(self, camera_id: str) -> None:
        """Stop and forget about one camera feed."""
        worker = self.workers.pop(camera_id, None)
        if worker:
            worker.stop()
            worker.join(timeout=5)
        self.states.pop(camera_id, None)

    def get_state(self, camera_id: str) -> Optional[CameraState]:
        """Read the latest state (frame + detections) for one camera."""
        return self.states.get(camera_id)

    def get_all_detections(self) -> Dict[str, List[Detection]]:
        """
        Convenience method for a downstream consumer (e.g. the
        distance/time-to-collision module) that wants "everything every
        camera currently sees", keyed by camera_id.
        """
        return {cam_id: state.last_detections for cam_id, state in self.states.items()}

    def shutdown(self) -> None:
        """Stop every camera worker cleanly - call this before the process exits."""
        for camera_id in list(self.workers.keys()):
            self.remove_camera(camera_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick standalone test of the multi-camera stream engine.")
    parser.add_argument(
        "--sources", nargs="+", default=["0"],
        help='One or more sources: RTSP URLs, video file paths, or webcam indices. '
             'Example: --sources 0  OR  --sources "rtsp://cam1/stream" "rtsp://cam2/stream"',
    )
    parser.add_argument("--conf", type=float, default=config.DEPLOY_CONFIDENCE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seconds", type=int, default=15, help="How long to run this demo for.")
    return parser.parse_args()


if __name__ == "__main__":
    """
    A minimal command-line demo: connect to one or more sources, run for
    a fixed number of seconds, and print detections as they come in.
    This proves the multi-camera engine works before wiring it into the
    full API server in app.py.
    """
    args = parse_args()

    manager = StreamManager(conf=args.conf, device=args.device)

    for i, source in enumerate(args.sources):
        # Webcam indices arrive as strings from argparse ("0"); convert
        # back to int so OpenCV interprets them as a device index rather
        # than (incorrectly) as a file path called "0".
        resolved_source = int(source) if source.isdigit() else source
        manager.add_camera(camera_id=f"camera_{i}", source=resolved_source)

    try:
        end_time = time.time() + args.seconds
        while time.time() < end_time:
            for camera_id, state in manager.states.items():
                if state.last_detections:
                    names = [f"{d.class_name}({d.confidence:.2f})" for d in state.last_detections]
                    print(f"[{camera_id}] fps={state.fps:.1f} detections={names}")
            time.sleep(1.0)
    finally:
        manager.shutdown()
