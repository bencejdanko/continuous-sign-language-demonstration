"""
inference_server.py — Translation model server for the AGX Orin demo.

Endpoints:
  GET  /health          → {"status": "ok", "model_loaded": bool}
  POST /translate       → body: JSON {"frames": [per-frame landmark dicts]}
                          response: {"translation": str, "num_windows": int}
  POST /reload          → re-downloads weights from HF and hot-reloads models

Usage:
  HF_TOKEN=<token> python inference_server.py

The server loads on startup:
  - SemanticEncoder  from bdanko/continuous-sign-language-translation/semantic_encoder.pth
  - TranslationModel from bdanko/continuous-sign-language-translation/translation_model.pth
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure models and data utilities are importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inference")

# ── Configuration ─────────────────────────────────────────────────────────────
HF_REPO   = os.environ.get("HF_REPO",   "bdanko/continuous-sign-language-translation")
HF_TOKEN  = os.environ.get("HF_TOKEN",  None)
PORT      = int(os.environ.get("INFERENCE_PORT", "8001"))
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model state (hot-reloadable) ──────────────────────────────────────────────
state = {
    "encoder":   None,
    "translator": None,
    "tokenizer": None,
    "loaded":    False,
}


def load_models():
    """Download weights from HF and load into GPU/CPU."""
    from huggingface_hub import hf_hub_download
    from transformers import T5Tokenizer
    from models import SemanticEncoder, TranslationModel

    log.info(f"Loading models from {HF_REPO} onto {DEVICE}...")

    enc_path = hf_hub_download(
        repo_id=HF_REPO, filename="semantic_encoder.pth", token=HF_TOKEN
    )
    trans_path = hf_hub_download(
        repo_id=HF_REPO, filename="translation_model.pth", token=HF_TOKEN
    )

    encoder = SemanticEncoder().to(DEVICE)
    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    translator = TranslationModel().to(DEVICE)
    translator.load_state_dict(torch.load(trans_path, map_location=DEVICE))
    translator.eval()

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

    state["encoder"]    = encoder
    state["translator"] = translator
    state["tokenizer"]  = tokenizer
    state["loaded"]     = True

    log.info("Models loaded ✓")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CSL Inference Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class LandmarkFrame(BaseModel):
    pose:  list  # [[{x,y,z}×33]]
    face:  list  # [[{x,y,z}×468]]
    hands: list  # [[{x,y,z}×21], ...]   (0, 1, or 2 hands)


class TranslateRequest(BaseModel):
    frames: list[LandmarkFrame]


class TranslateResponse(BaseModel):
    translation: str
    num_windows: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": state["loaded"]}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not state["loaded"]:
        raise HTTPException(503, "Models not loaded yet")

    from data import landmarks_dict_to_array, engineer_features, sliding_windows

    # Convert request frames → [T, 543, 3]
    frame_dicts = [f.model_dump() for f in req.frames]
    raw = landmarks_dict_to_array(frame_dicts)
    if raw is None or raw.shape[0] < 2:
        raise HTTPException(400, "Not enough landmark frames (need ≥ 2)")

    # Feature engineering → [T, 540]
    features = engineer_features(raw)
    if features is None:
        raise HTTPException(400, "Feature engineering failed")

    encoder    = state["encoder"]
    translator = state["translator"]
    tokenizer  = state["tokenizer"]

    translations = []
    with torch.no_grad():
        for chunk in sliding_windows(features):          # [60, 540]
            x = chunk.unsqueeze(0).to(DEVICE)            # [1, 60, 540]
            z = encoder(x)                               # [1, 512, 15]
            z_seq = z.transpose(1, 2)                    # [1, 15, 512]
            ids = translator(z_seq)                      # [1, seq_len]
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            if text.strip():
                translations.append(text.strip())

    # Aggregate: join unique window translations
    seen = set()
    unique = []
    for t in translations:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    final = " | ".join(unique) if unique else "[no translation]"
    return TranslateResponse(translation=final, num_windows=len(translations))


@app.post("/reload")
def reload_models():
    """Hot-reload weights from HF (call this after uploading a new checkpoint)."""
    try:
        load_models()
        return {"status": "reloaded", "device": DEVICE}
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_server:app", host="0.0.0.0", port=PORT, reload=False)
