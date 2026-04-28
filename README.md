# AGX Orin — Deployment Guide
## Continuous Sign Language Translation Demo

---

## System Overview

```
┌──────────────────────────────────────────────────────┐
│  AGX Orin                                            │
│                                                      │
│  ┌────────────────┐      ┌─────────────────────────┐ │
│  │   app.py       │ POST │  inference_server.py    │ │
│  │  (port 8000)   │─────▶│  (port 8001)            │ │
│  │                │      │                         │ │
│  │ • Raw preview  │      │ • SemanticEncoder       │ │
│  │ • Record clips │      │ • FLAN-T5-small         │ │
│  │ • MediaPipe    │      │ • /translate            │ │
│  │ • LM replay    │      │ • /reload               │ │
│  └──────┬─────────┘      └─────────────────────────┘ │
│         │ WebSocket                                   │
└─────────┼────────────────────────────────────────────┘
          │  Browser (any device on LAN)
          ▼
    http://<orin-ip>:8000
```

**Workflow:**
1. User opens browser → sees live camera feed
2. Clicks **Record** → raw frames are buffered on the Orin
3. Clicks **Translate** → MediaPipe runs on the buffer in one batch → landmark array sent to inference server → FLAN-T5 decodes → translation displayed
4. Clicks **Replay** → annotated video (with full landmark skeleton) streamed back

---

## Prerequisites

- NVIDIA Jetson AGX Orin with **JetPack 5.x or 6.x**
- Python **3.10+** (shipped with JetPack 5+)
- USB / CSI camera
- Internet access for first-time model download

---

## 1 — Install PyTorch for Jetson

> [!IMPORTANT]
> Do **not** `pip install torch` directly — it will pull the x86 build. Use NVIDIA's Jetson wheel.

```bash
# Check your JetPack version
dpkg -l | grep nvidia-jetpack

# JetPack 6.x  (L4T R36)
pip install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# JetPack 5.x  (L4T R35)  — adjust version as needed
pip install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.x.x  True
```

> [!TIP]
> The full list of Jetson PyTorch wheels is at:
> https://developer.nvidia.com/embedded/downloads#?tx=$product,jetson_agx_orin

---

## 2 — Install Python dependencies

```bash
pip install \
  fastapi "uvicorn[standard]" httpx \
  mediapipe opencv-python \
  transformers sentencepiece \
  huggingface_hub \
  datasets \
  numpy pydantic
```

> [!NOTE]
> `mediapipe` ships pre-compiled wheels for aarch64 since v0.10.
> If your version doesn't have an aarch64 wheel, install from source:
> https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md

---

## 3 — Clone the repo

```bash
git clone https://github.com/bencejdanko/continuous-sign-language-translation
cd continuous-sign-language-translation
chmod +x start.sh
```

---

## 4 — Set your Hugging Face token

The inference l model weights from `bdanko/continuous-sign-language-translation` on startup.

```bash
export HF_TOKEN="hf_your_token_here"
# Persist across reboots:
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

---

## 5 — Run the demo

```bash
# Start both servers (blocks; Ctrl-C to stop)
HF_TOKEN=$HF_TOKEN ./start.sh

# With a non-default camera
CAM_INDEX=1 HF_TOKEN=$HF_TOKEN ./start.sh
```

Open `http://<orin-ip>:8000` in a browser on any machine on the same network.

---

## 6 — Update model weights after a new training run

After you run Colab Phase 1 or Phase 2 and new weights are uploaded to HF:

```bash
# Hot-reload without restarting servers
curl -X POST http://localhost:8001/reload
```

This re-downloads `semantic_encoder.pth` and `translation_model.pth` from HF and swaps them in-place with no downtime.

---

## 7 — Autostart with systemd (optional)

Create `/etc/systemd/system/csl-demo.service`:

```ini
[Unit]
Description=Continuous Sign Language Translation Demo
After=network.target

[Service]
Type=forking
User=YOUR_USERNAME
WorkingDirectory=/path/to/continuous-sign-language-translation
Environment="HF_TOKEN=hf_your_token_here"
Environment="CAM_INDEX=0"
ExecStart=/bin/bash /path/to/continuous-sign-language-translation/start.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable csl-demo
sudo systemctl start csl-demo
```

---

## 8 — Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` returns `False` | Installed wrong wheel — re-install Jetson wheel from §1 |
| `mediapipe` import error | `pip install --upgrade mediapipe` or build from source |
| `httpx.ConnectError` on translate | Inference server isn't ready — check `curl http://localhost:8001/health` |
| Translation is `[no translation]` | Model is freshly initialised (run full Colab training first) |
| Camera not found | Try `CAM_INDEX=1` or check `ls /dev/video*` |
| OOM on `translation_model.pth` (308 MB) | Ensure ≥8 GB unified memory; close other GPU apps |

---

## 9 — File reference

| File | Purpose |
|---|---|
| `app.py` | Demo frontend — camera, record, MediaPipe, WS |
| `inference_server.py` | Model server — `/translate`, `/reload`, `/health` |
| `data.py` | Feature engineering shared by training and inference |
| `models.py` | `SemanticEncoder`, `DiffusionDecoder`, `TranslationModel` |
| `start.sh` | Starts both servers; waits for inference health check |
| `colab_phase1_diffusion.ipynb` | Phase 1 training notebook (Colab) |
| `colab_phase2_translation.ipynb` | Phase 2 translation training notebook (Colab) |
