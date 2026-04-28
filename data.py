"""
data.py — Feature engineering for Continuous Sign Language Translation.

Landmark layout in the raw 543-pt MediaPipe Holistic array:
  [  0 –  32]  33  Pose landmarks
  [ 33 – 500]  468 Face mesh landmarks
  [501 – 521]  21  Left-hand landmarks
  [522 – 542]  21  Right-hand landmarks

engineer_features():  [T, 543, 3]  →  torch.Tensor [T, 540]
sliding_windows():    [T, 540]     →  iterator of [T_WINDOW, 540] chunks

RealSignLanguageDataset: streams from HF, applies engineering on-the-fly.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

# 15 face landmarks capturing Non-Manual Markers critical for ASL grammar.
# Indices are relative to the face sub-array (0-467).
FACE_LANDMARK_IDXS = [
    70,  105,  # left eyebrow inner / peak
    336, 300,  # right eyebrow peak / inner
    33,  133,  # left eye corners
    362, 263,  # right eye corners
    4,         # nose tip
    61,  291,  # lip corners
    13,  14,   # upper / lower lip centre
    17,        # chin
    0,         # mouth top centre
]
assert len(FACE_LANDMARK_IDXS) == 15

_POSE  = slice(0,   33)
_FACE  = slice(33,  501)
_LHAND = slice(501, 522)
_RHAND = slice(522, 543)

T_WINDOW = 60
T_STRIDE = 30


def engineer_features(raw: np.ndarray) -> torch.Tensor | None:
    """
    raw: np.ndarray [T, 543, 3]
    Returns torch.Tensor [T, 540]  or None if too short to process.

    Pipeline:
        1. Spatial normalisation  – subtract pose centroid per frame
        2. Face downsampling      – 468 → 15 key landmarks
        3. Concatenate            – 33 + 15 + 21 + 21 = 90 keypoints
        4. Temporal deltas        – Δx Δy Δz  (frame-to-frame diff)
        5. Flatten + concatenate  – [T, 90×3] + [T, 90×3] = [T, 540]
    """
    T = raw.shape[0]
    if T < 2:
        return None

    pose  = raw[:, _POSE,  :]   # [T, 33, 3]
    face  = raw[:, _FACE,  :]   # [T, 468, 3]
    lhand = raw[:, _LHAND, :]   # [T, 21, 3]
    rhand = raw[:, _RHAND, :]   # [T, 21, 3]

    # Body-centred normalisation
    centre = pose.mean(axis=1, keepdims=True)   # [T, 1, 3]
    pose, face, lhand, rhand = (a - centre for a in (pose, face, lhand, rhand))

    # Assemble 90 keypoints
    kpts = np.concatenate(
        [pose, face[:, FACE_LANDMARK_IDXS, :], lhand, rhand], axis=1
    )  # [T, 90, 3]

    # Temporal deltas (first frame delta = 0)
    delta = np.zeros_like(kpts)
    delta[1:] = kpts[1:] - kpts[:-1]

    # Flatten → [T, 540]
    features = np.concatenate(
        [kpts.reshape(T, -1), delta.reshape(T, -1)], axis=1
    ).astype(np.float32)

    return torch.from_numpy(features)


def sliding_windows(
    feat: torch.Tensor,
    window: int = T_WINDOW,
    stride: int = T_STRIDE,
):
    """Yield [window, F] chunks.  Short clips are zero-padded to one window."""
    T = feat.shape[0]
    if T < window:
        yield F.pad(feat, (0, 0, 0, window - T))
    else:
        start = 0
        while start + window <= T:
            yield feat[start : start + window]
            start += stride


def landmarks_dict_to_array(data: dict) -> np.ndarray | None:
    """
    Convert the JSON landmark dict produced by app.py into a [T, 543, 3]
    float32 array ready for engineer_features().

    data format (per frame):
        {"pose": [[{x,y,z}×33]], "face": [[{x,y,z}×468]], "hands": [[{x,y,z}×21], ...]}

    Returns None if any required group is missing on a frame.
    """
    T = len(data)
    if T == 0:
        return None

    out = np.zeros((T, 543, 3), dtype=np.float32)
    for i, frame in enumerate(data):
        pose  = frame.get("pose",  [[]])
        face  = frame.get("face",  [[]])
        hands = frame.get("hands", [])

        pose_pts  = pose[0]  if pose  else []
        face_pts  = face[0]  if face  else []
        lhand_pts = hands[0] if len(hands) > 0 else []
        rhand_pts = hands[1] if len(hands) > 1 else []

        def fill(pts, start, count):
            for j, pt in enumerate(pts[:count]):
                out[i, start + j, 0] = pt["x"]
                out[i, start + j, 1] = pt["y"]
                out[i, start + j, 2] = pt["z"]

        fill(pose_pts,  0,   33)
        fill(face_pts,  33,  468)
        fill(lhand_pts, 501, 21)
        fill(rhand_pts, 522, 21)

    return out


class RealSignLanguageDataset(IterableDataset):
    """
    Streams from `bdanko/how2sign-landmarks-front-raw-parquet` on Hugging Face.
    Applies on-the-fly feature engineering (normalise → downsample → deltas).

    Args:
        split:       "train" | "validation" | "test"
        max_samples: cap raw clips streamed (None = full dataset)
        repo_id:     HF dataset repo
    """
    def __init__(
        self,
        split: str = "train",
        max_samples: int | None = None,
        repo_id: str = "bdanko/how2sign-landmarks-front-raw-parquet",
    ):
        self.split = split
        self.max_samples = max_samples
        self.repo_id = repo_id

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        count = 0
        for sample in ds:
            if self.max_samples is not None and count >= self.max_samples:
                break
            raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
            feat = engineer_features(raw)
            if feat is None:
                continue
            for chunk in sliding_windows(feat):
                yield chunk, sample.get("sentence", "")
            count += 1
