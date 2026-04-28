"""
app.py — Demo frontend for Continuous Sign Language Translation.

Architecture:
  ┌────────────────────────────────────────────┐
  │  Browser                                   │
  │   ├── Live raw camera preview (JPEG WS)    │
  │   ├── [Record] → buffers raw frames        │
  │   ├── [Translate] → MediaPipe batch-run    │
  │   │    → POST /translate to inference_srv  │
  │   │    → display translation text          │
  │   └── [Replay] → landmark-annotated video  │
  └────────────────────────────────────────────┘

Key design choices for AGX Orin:
  - No landmark rendering during live preview  → lower latency / bandwidth
  - MediaPipe runs ONCE on the buffered clip   → clean batch input for the model
  - Landmark replay is generated server-side   → browser just plays JPEG frames
  - inference_server runs on the same machine  → localhost POST, no network hop

Environment variables:
  CAM_INDEX         camera device index           (default 0)
  INFERENCE_PORT    port of the inference server  (default 8001)
"""

import asyncio
import json
import os
import time
import urllib.request
from contextlib import asynccontextmanager

import cv2
import httpx
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
CAM_INDEX  = int(os.environ.get("CAM_INDEX", "0"))
CAM_W, CAM_H = 640, 480
JPEG_Q     = 70                   # quality for raw preview stream
INFER_URL  = f"http://localhost:{os.environ.get('INFERENCE_PORT', '8001')}/translate"

MEDIAPIPE_MODELS = {
    "pose_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
}

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),(0,1),(1,2),(2,3),(3,7),(0,4),
    (4,5),(5,6),(6,8),(9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
    (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
    (18,19),(19,20),(5,9),(9,13),(13,17),
]

# ── Global state ──────────────────────────────────────────────────────────────
cap         = None
pose_det    = None
face_det    = None
hand_det    = None
is_recording = False
frame_buffer: list[np.ndarray] = []   # raw BGR frames while recording


def _download_mp_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, url in MEDIAPIPE_MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}…")
            urllib.request.urlretrieve(url, path)


def _mp_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)


# ── MediaPipe helpers ─────────────────────────────────────────────────────────

def _run_mediapipe_on_frames(frames: list[np.ndarray]) -> list[dict]:
    """
    Runs pose / face / hand landmarkers on a list of BGR frames.
    Returns one landmark-dict per frame (suitable for the inference server).
    """
    # Use IMAGE mode so we don't have to worry about monotonic timestamps
    pose_img = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_mp_path("pose_landmarker.task")),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )
    face_img = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_mp_path("face_landmarker.task")),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
    )
    hand_img = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_mp_path("hand_landmarker.task")),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )

    results = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pr = pose_img.detect(mp_img)
        fr = face_img.detect(mp_img)
        hr = hand_img.detect(mp_img)

        def lm_list(result_obj, attr):
            out = []
            for group in (getattr(result_obj, attr, None) or []):
                out.append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in group])
            return out

        results.append({
            "pose":  lm_list(pr, "pose_landmarks"),
            "face":  lm_list(fr, "face_landmarks"),
            "hands": lm_list(hr, "hand_landmarks"),
        })

    pose_img.close(); face_img.close(); hand_img.close()
    return results


def _draw_landmarks(bgr: np.ndarray, lm_frame: dict) -> np.ndarray:
    h, w = bgr.shape[:2]
    out = bgr.copy()

    # Pose
    for group in lm_frame.get("pose", []):
        pts = [(int(p["x"]*w), int(p["y"]*h)) for p in group]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], (0, 220, 0), 2)
        for pt in pts:
            cv2.circle(out, pt, 4, (0, 255, 0), -1)

    # Face
    for group in lm_frame.get("face", []):
        for p in group:
            cv2.circle(out, (int(p["x"]*w), int(p["y"]*h)), 1, (0, 165, 255), -1)

    # Hands
    for group in lm_frame.get("hands", []):
        hpts = [(int(p["x"]*w), int(p["y"]*h)) for p in group]
        for a, b in HAND_CONNECTIONS:
            if a < len(hpts) and b < len(hpts):
                cv2.line(out, hpts[a], hpts[b], (60, 60, 255), 2)
        for pt in hpts:
            cv2.circle(out, pt, 5, (80, 80, 255), -1)

    return out


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cap, pose_det, face_det, hand_det
    _download_mp_models()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # Lightweight landmarkers for the live overlay (VIDEO mode, runs every N frames)
    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_mp_path("pose_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )
    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_mp_path("face_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_mp_path("hand_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )

    print(f"Camera index={CAM_INDEX}, opened={cap.isOpened()}")
    print(f"Inference server: {INFER_URL}")
    yield

    cap.release()
    pose_det.close(); face_det.close(); hand_det.close()


app = FastAPI(title="CSL Demo", lifespan=lifespan)


# ── HTML page ─────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sign Language Translation Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 2rem; }
  h1 { font-size: 1.6rem; margin-bottom: 1.5rem; color: #58a6ff; letter-spacing: .04em; }
  #video-wrap { position: relative; width: 640px; }
  canvas { width: 640px; height: 480px; border-radius: 8px;
           border: 2px solid #21262d; display: block; }
  #overlay-label { position: absolute; top: 8px; left: 12px;
                   background: rgba(0,0,0,.55); padding: 3px 8px; border-radius: 4px;
                   font-size: .75rem; color: #8b949e; }
  #controls { display: flex; gap: .75rem; margin-top: 1rem; flex-wrap: wrap; justify-content: center; }
  button { padding: .55rem 1.4rem; border: none; border-radius: 6px; cursor: pointer;
           font-size: .9rem; font-weight: 600; transition: opacity .15s; }
  button:disabled { opacity: .35; cursor: default; }
  #btn-record  { background: #238636; color: #fff; }
  #btn-stop    { background: #da3633; color: #fff; }
  #btn-translate { background: #1f6feb; color: #fff; }
  #btn-replay  { background: #6e40c9; color: #fff; }
  #status { margin-top: 1rem; font-size: .85rem; color: #8b949e; min-height: 1.2em; }
  #result { margin-top: 1.2rem; width: 640px; background: #161b22;
            border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.2rem;
            min-height: 4rem; font-size: 1.05rem; line-height: 1.6; }
  #result .label { font-size: .7rem; color: #8b949e; text-transform: uppercase;
                   letter-spacing: .08em; margin-bottom: .4rem; }
  #result .text { color: #e6edf3; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 6px; }
  .dot.green { background: #3fb950; }
  .dot.red   { background: #f85149; }
  .dot.grey  { background: #6e7681; }
</style>
</head>
<body>
<h1>🤟 Continuous Sign Language Translation</h1>

<div id="video-wrap">
  <canvas id="canvas"></canvas>
  <div id="overlay-label"><span class="dot grey" id="status-dot"></span><span id="mode-label">Connecting…</span></div>
</div>

<div id="controls">
  <button id="btn-record"    disabled>⏺ Record</button>
  <button id="btn-stop"      disabled>⏹ Stop</button>
  <button id="btn-translate" disabled>🔤 Translate</button>
  <button id="btn-replay"    disabled>▶ Replay Landmarks</button>
</div>
<div id="status"></div>

<div id="result">
  <div class="label">Translation</div>
  <div class="text" id="trans-text">—</div>
</div>

<script>
const canvas   = document.getElementById("canvas");
const ctx      = canvas.getContext("2d");
canvas.width   = 640; canvas.height = 480;

const btnRecord    = document.getElementById("btn-record");
const btnStop      = document.getElementById("btn-stop");
const btnTranslate = document.getElementById("btn-translate");
const btnReplay    = document.getElementById("btn-replay");
const statusEl     = document.getElementById("status");
const modeLabel    = document.getElementById("mode-label");
const statusDot    = document.getElementById("status-dot");
const transText    = document.getElementById("trans-text");

let ws, isRecording = false, hasBuffer = false;

function setStatus(msg)   { statusEl.textContent = msg; }
function setMode(msg, col) {
  modeLabel.textContent = msg;
  statusDot.className = "dot " + col;
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/demo`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setMode("Live preview", "green");
    btnRecord.disabled = false;
  };

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // JPEG frame
      const blob = new Blob([ev.data], {type:"image/jpeg"});
      const url  = URL.createObjectURL(blob);
      const img  = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0); URL.revokeObjectURL(url); };
      img.src = url;
    } else {
      const msg = JSON.parse(ev.data);
      if (msg.type === "recording_started") {
        setMode("Recording…", "red");
        isRecording = true; hasBuffer = false;
        btnRecord.disabled    = true;
        btnStop.disabled      = false;
        btnTranslate.disabled = true;
        btnReplay.disabled    = true;
      } else if (msg.type === "recording_stopped") {
        setMode("Buffer ready", "grey");
        isRecording = false; hasBuffer = true;
        btnStop.disabled      = true;
        btnRecord.disabled    = false;
        btnTranslate.disabled = false;
        btnReplay.disabled    = false;
        setStatus(`Captured ${msg.frames} frames — click Translate or Replay.`);
      } else if (msg.type === "processing") {
        setStatus(msg.message);
        btnTranslate.disabled = true;
      } else if (msg.type === "translation") {
        transText.textContent = msg.text;
        setStatus(`Done. ${msg.num_windows} window(s) processed.`);
        btnTranslate.disabled = false;
        setMode("Live preview", "green");
      } else if (msg.type === "error") {
        setStatus("Error: " + msg.message);
        btnTranslate.disabled = false;
        setMode("Live preview", "green");
      }
    }
  };

  ws.onclose = () => {
    setMode("Disconnected", "grey");
    setTimeout(connect, 2000);
  };
}

btnRecord.onclick = () => {
  ws.send(JSON.stringify({cmd: "start_recording"}));
};
btnStop.onclick = () => {
  ws.send(JSON.stringify({cmd: "stop_recording"}));
};
btnTranslate.onclick = () => {
  ws.send(JSON.stringify({cmd: "translate"}));
  setStatus("Running MediaPipe + translation…");
};
btnReplay.onclick = () => {
  ws.send(JSON.stringify({cmd: "replay_landmarks"}));
  setStatus("Streaming landmark replay…");
};

connect();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ── Demo WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/ws/demo")
async def demo_ws(ws: WebSocket):
    """
    Single WebSocket that handles the full demo lifecycle:
      - Continuously streams raw camera JPEGs (binary messages)
      - Listens for control commands (text JSON messages)
      - On "translate": runs MediaPipe on buffer, calls inference server, replies
      - On "replay_landmarks": streams annotated-JPEG replay (binary)
    """
    global is_recording, frame_buffer
    await ws.accept()

    frame_count = 0
    pose_res = face_res = hand_res = None
    OVERLAY_EVERY = 5   # run lightweight pose overlay every N preview frames

    async def send_json(obj: dict):
        await ws.send_text(json.dumps(obj))

    try:
        while True:
            # ── Check for incoming commands (non-blocking) ──
            try:
                raw_msg = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                cmd = json.loads(raw_msg).get("cmd")

                if cmd == "start_recording":
                    is_recording = True
                    frame_buffer.clear()
                    await send_json({"type": "recording_started"})

                elif cmd == "stop_recording":
                    is_recording = False
                    await send_json({
                        "type": "recording_stopped",
                        "frames": len(frame_buffer),
                    })

                elif cmd == "translate":
                    if not frame_buffer:
                        await send_json({"type": "error", "message": "No recorded frames."})
                    else:
                        await send_json({
                            "type": "processing",
                            "message": f"Running MediaPipe on {len(frame_buffer)} frames…",
                        })
                        loop = asyncio.get_event_loop()

                        # Run MediaPipe in thread pool to avoid blocking the event loop
                        lm_frames = await loop.run_in_executor(
                            None, _run_mediapipe_on_frames, list(frame_buffer)
                        )

                        await send_json({
                            "type": "processing",
                            "message": "Calling translation model…",
                        })

                        try:
                            async with httpx.AsyncClient(timeout=60.0) as client:
                                resp = await client.post(
                                    INFER_URL, json={"frames": lm_frames}
                                )
                                resp.raise_for_status()
                                result = resp.json()
                            await send_json({
                                "type": "translation",
                                "text": result["translation"],
                                "num_windows": result["num_windows"],
                            })
                        except Exception as e:
                            await send_json({"type": "error", "message": str(e)})

                elif cmd == "replay_landmarks":
                    if not frame_buffer:
                        await send_json({"type": "error", "message": "No recorded frames."})
                    else:
                        await send_json({
                            "type": "processing",
                            "message": f"Rendering landmarks on {len(frame_buffer)} frames…",
                        })
                        loop = asyncio.get_event_loop()
                        lm_frames = await loop.run_in_executor(
                            None, _run_mediapipe_on_frames, list(frame_buffer)
                        )
                        for bgr, lmf in zip(frame_buffer, lm_frames):
                            ann = _draw_landmarks(bgr, lmf)
                            _, jpg = cv2.imencode(
                                ".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]
                            )
                            await ws.send_bytes(jpg.tobytes())
                            await asyncio.sleep(1 / 30)   # ~30 fps replay

            except asyncio.TimeoutError:
                pass   # no command this tick — continue to frame send

            # ── Capture + stream live preview frame ───────────────────────────
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            if is_recording:
                frame_buffer.append(frame.copy())

            frame_count += 1
            _, jpg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]
            )
            await ws.send_bytes(jpg.tobytes())
            await asyncio.sleep(0)   # yield to event loop

    except WebSocketDisconnect:
        is_recording = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)