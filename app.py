"""
app.py
=======
The backend server that runs on the CENTRAL WORKSTATION. This is the
"product" of the hackathon submission - the thing a judge (or a
teammate's downstream module) actually talks to, as opposed to the
training notebook, which is documentation of how the model was built.

WHAT THIS SERVER DOES
------------------------
1. On startup, loads the trained model ONCE and creates a
   `StreamManager` (from src/infer_stream.py) that can watch several
   camera feeds at once.
2. Exposes a small HTTP API:
     GET  /health                     - is the server alive?
     POST /cameras/{camera_id}        - start watching a new camera feed
     DELETE /cameras/{camera_id}      - stop watching a camera feed
     GET  /cameras                    - list currently-watched cameras + their status
     GET  /cameras/{camera_id}/detections   - latest detections for one camera (JSON)
     GET  /cameras/{camera_id}/frame        - latest annotated frame, as a JPEG image
     POST /infer/image                - upload a single image, get detections back (no camera needed)

WHY FASTAPI?
--------------
FastAPI is a lightweight Python web framework that's fast to write,
auto-generates interactive API docs (visit /docs once running), and
plays nicely with type hints - a good fit for a hackathon backend that
other teammates (e.g. whoever builds the distance/TTC module, or a
frontend dashboard) need to integrate with quickly.

HOW TO RUN
-----------
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs in a browser for interactive docs,
or http://localhost:8000/health to confirm it's running.

NOTE ON THE WEBCAM QUICK-DEMO ENDPOINT
------------------------------------------
For a hackathon demo without real CCTV hardware, you can register your
laptop's webcam as "camera_0" by POSTing source="0" to /cameras/camera_0
- see the README for a copy-pasteable example.
"""

import io
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from ultralytics import YOLO

import config
from src.infer_stream import StreamManager

app = FastAPI(
    title="Fog-Mine Safety Detection Service",
    description="Central inference service: human/vehicle/obstacle detection over CCTV feeds.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global state, created once when the server starts.
# ---------------------------------------------------------------------------
# In a small hackathon-scale service, a couple of module-level globals
# for shared state is a reasonable, simple choice. In a larger
# production system you'd likely manage this via FastAPI's dependency
# injection or an application state object instead.
stream_manager: Optional[StreamManager] = None
single_image_model: Optional[YOLO] = None


@app.on_event("startup")
def on_startup() -> None:
    """
    Runs once, when the server process starts (before it accepts any
    requests). We load the model(s) here rather than per-request, since
    loading a YOLO model from disk is relatively slow - we want to pay
    that cost exactly once, not on every incoming HTTP request.
    """
    global stream_manager, single_image_model

    config.ensure_directories()

    print("Loading model for the multi-camera stream manager...")
    stream_manager = StreamManager(
        weights_path=str(config.BEST_MODEL_PATH),
        conf=config.DEPLOY_CONFIDENCE,
        device="cpu",   # change to 0 if the central workstation has a CUDA GPU
    )

    print("Loading model for single-image inference...")
    single_image_model = YOLO(str(config.BEST_MODEL_PATH))

    print("Server ready.")


@app.on_event("shutdown")
def on_shutdown() -> None:
    """Cleanly stop every camera worker thread when the server is shutting down."""
    if stream_manager:
        stream_manager.shutdown()


# ---------------------------------------------------------------------------
# Request/response schemas (Pydantic models)
# ---------------------------------------------------------------------------
# Defining these gives FastAPI enough information to validate incoming
# requests automatically and to generate the interactive /docs page -
# you get input validation "for free" just by writing normal-looking
# Python classes with type hints.

class AddCameraRequest(BaseModel):
    # "source" is deliberately a string even for webcam indices (e.g.
    # "0") - we convert to int inside the endpoint, since JSON request
    # bodies don't need to distinguish "0 the webcam index" from
    # "0 the string" at this layer.
    source: str
    target_fps: Optional[float] = None


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple liveness check - useful for a monitoring dashboard or a load balancer."""
    return {"status": "ok", "classes": config.CLASS_NAMES}


@app.post("/cameras/{camera_id}")
def add_camera(camera_id: str, request: AddCameraRequest):
    """
    Start watching a new camera feed.

    Example (from a terminal, using curl):
        curl -X POST http://localhost:8000/cameras/entrance_cam \\
             -H "Content-Type: application/json" \\
             -d '{"source": "rtsp://192.168.1.10:554/stream1"}'

    For a quick demo using your own laptop webcam instead of real CCTV:
        curl -X POST http://localhost:8000/cameras/camera_0 \\
             -H "Content-Type: application/json" \\
             -d '{"source": "0"}'
    """
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    # Convert "0", "1" style strings back into real ints so OpenCV
    # treats them as a local webcam device index, not a filename.
    resolved_source = int(request.source) if request.source.isdigit() else request.source

    try:
        stream_manager.add_camera(camera_id, resolved_source, target_fps=request.target_fps)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"camera_id": camera_id, "source": request.source, "status": "starting"}


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    """Stop watching a camera feed and free its resources."""
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    stream_manager.remove_camera(camera_id)
    return {"camera_id": camera_id, "status": "removed"}


@app.get("/cameras")
def list_cameras():
    """List every currently-registered camera and its connection status."""
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    return {
        camera_id: {
            "connected": state.connected,
            "fps": round(state.fps, 1),
            "last_update_time": state.last_update_time,
            "num_current_detections": len(state.last_detections),
            "error": state.error,
        }
        for camera_id, state in stream_manager.states.items()
    }


@app.get("/cameras/{camera_id}/detections", response_model=list[DetectionResponse])
def get_camera_detections(camera_id: str):
    """
    The main endpoint a DOWNSTREAM MODULE (e.g. the distance/
    time-to-collision code your teammates are building) would poll: the
    latest set of detected objects for one camera, as plain JSON, with
    pixel bounding boxes ready to feed into further geometry
    calculations.
    """
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    state = stream_manager.get_state(camera_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id: {camera_id}")

    return [
        DetectionResponse(
            class_id=d.class_id, class_name=d.class_name, confidence=d.confidence,
            x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
        )
        for d in state.last_detections
    ]


@app.get("/cameras/{camera_id}/frame")
def get_camera_frame(camera_id: str):
    """
    Returns the latest ANNOTATED frame (boxes already drawn on it) for
    one camera, as a JPEG image - useful for a live web dashboard
    (`<img src="/cameras/entrance_cam/frame">` refreshed periodically)
    or for a quick manual check during the hackathon demo.
    """
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    state = stream_manager.get_state(camera_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id: {camera_id}")

    if state.last_annotated_frame is None:
        raise HTTPException(status_code=404, detail="No frame received yet from this camera.")

    # cv2.imencode compresses the raw NumPy frame into JPEG bytes in
    # memory (no temp file needed) - `.tobytes()` gives us the raw bytes
    # to send back over HTTP.
    success, buffer = cv2.imencode(".jpg", state.last_annotated_frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/infer/image", response_model=list[DetectionResponse])
async def infer_single_image(file: UploadFile = File(...)):
    """
    Run inference on a SINGLE uploaded image (not a live camera feed) -
    useful for testing the model from a browser, a Postman request, or
    a quick "upload a photo" demo screen in your hackathon presentation.
    """
    if single_image_model is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    # Read the uploaded file's bytes and decode them into an OpenCV
    # image (a NumPy array), entirely in memory - no need to save the
    # upload to disk first.
    contents = await file.read()
    import numpy as np
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded file as an image.")

    results = single_image_model.predict(
        source=frame, imgsz=config.IMAGE_SIZE, conf=config.DEPLOY_CONFIDENCE, verbose=False,
    )
    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(DetectionResponse(
                class_id=cls_id,
                class_name=config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else f"class_{cls_id}",
                confidence=float(box.conf.item()),
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            ))

    return detections


# ---------------------------------------------------------------------------
# Running this file directly (`python app.py`) also works, using uvicorn's
# programmatic API, as an alternative to the `uvicorn app:app` CLI command
# shown in the module docstring above.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
