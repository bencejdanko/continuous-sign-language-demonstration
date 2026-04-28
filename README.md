# AGX Orin — Deployment Guide
## Continuous Sign Language Translation Demo

<img width="1068" height="1040" alt="image" src="https://github.com/user-attachments/assets/d69554f8-d9b5-458d-a9b5-038788ea9d5b" />


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

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.x.x  True
```

> [!TIP]
> The full list of Jetson PyTorch wheels is at:
> https://developer.nvidia.com/embedded/downloads#?tx=$product,jetson_agx_orin

---

## 2 — Pull repo and Install Python dependencies

```bash
git clone https://github.com/bencejdanko/continuous-sign-language-demo-agx-orin

cd continuous-sign-language-demo-agx-orin

pip install -r inference_server_requirements.txt

chmod +x start.sh
```

---

## 4 — Set your Hugging Face token

The inference l model weights from on startup.

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
```

Open `http://<orin-ip>:8000` in a browser on any machine on the same network.

---

Then start the app server on your laptop/machine w/ a camera:

```bash
pip install -r app_server_requirements.txt

# windows
$env:CAM_INDEX="0"; $env:HOST="ubuntu"; $env:INFERENCE_PORT="8001"; python app.py

```
---

## 6 — Update model weights after a new training run

After you run Colab Phase 1 or Phase 2 and new weights are uploaded to HF:

```bash
# Hot-reload without restarting servers
curl -X POST http://localhost:8001/reload
```

This re-downloads `semantic_encoder.pth` and `translation_model.pth` from HF and swaps them in-place with no downtime.
