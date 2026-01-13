"""
Kalman Transformer NATS Server for Real-Time Fall Detection

This server uses the TRAINED StandardScaler from the training pipeline
to ensure feature normalization matches exactly what the model learned.

CRITICAL FIX: Replaced per-window normalization with trained scaler.
Root cause of false positives: per-window stats differ from training stats.

Required files in same directory:
- acc_scaler.pkl: StandardScaler fitted on training data
- best_model.pth: Trained model weights
- kalman_transformer_variants.py: Model architecture
- linear_kalman_orientations.py: Kalman filter implementation

Usage:
    export DEVICE=cpu
    export MODEL_PATH=best_model.pth
    export SCALER_PATH=acc_scaler.pkl
    python server_kalman_transformer_nats.py
"""

import asyncio
import json
import os
import time
from typing import Any, Tuple

import numpy as np
import torch
import joblib

from nats.aio.client import Client as NATS

from kalman_transformer_variants import KalmanBalancedFlexible
from linear_kalman_orientations import LinearKalmanFilter


# -----------------------------
# Env / Config
# -----------------------------
DEVICE = os.environ.get("DEVICE", "cpu")
DEFAULT_FS_HZ = float(os.environ.get("FS_HZ", "30.0"))
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pth")
SCALER_PATH = os.environ.get("SCALER_PATH", "acc_scaler.pkl")
NATS_URL = os.environ.get("NATS_URL", "nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224")

MODEL = None
ACC_SCALER = None  # Trained StandardScaler for accelerometer normalization


# -----------------------------
# Helpers
# -----------------------------
def _as_float32_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr


def _fix_nans(arr: np.ndarray) -> np.ndarray:
    """Replace NaNs with column mean (or 0 if whole column is NaN)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return arr

    # If an entire column is NaN -> set to 0
    col_all_nan = np.all(np.isnan(arr), axis=0)
    if np.any(col_all_nan):
        arr[:, col_all_nan] = 0.0

    # Mean-impute remaining NaNs
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        col_mean = np.nanmean(arr, axis=0)
        arr[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    return arr


def _ensure_128x3(x: np.ndarray, name: str) -> np.ndarray:
    """
    Expect [128][3] but allow slight mismatch:
    - if >128: take last 128 (most recent)
    - if <128: left-pad zeros
    """
    if x is None:
        raise ValueError(f"Missing '{name}' in payload")

    if x.ndim != 2:
        raise ValueError(f"'{name}' must be 2D array [T][F], got shape={x.shape}")

    if x.shape[1] < 3:
        raise ValueError(f"'{name}' must have at least 3 columns, got {x.shape[1]}")

    x = x[:, :3]  # keep xyz
    T = x.shape[0]
    TARGET = 128

    if T == TARGET:
        return x.astype(np.float32, copy=False)

    if T > TARGET:
        return x[-TARGET:, :].astype(np.float32, copy=False)

    pad = np.zeros((TARGET - T, 3), dtype=np.float32)
    return np.vstack([pad, x]).astype(np.float32, copy=False)


def _convert_gyro_units(gyro: np.ndarray, units: str) -> np.ndarray:
    """
    Kalman preprocessing expects rad/s. Convert if watch sends deg/s.
    """
    if not units:
        return gyro
    u = units.strip().lower()
    if u in ("deg/s", "dps", "degrees/s", "degrees_per_second"):
        return np.deg2rad(gyro).astype(np.float32)
    return gyro


def parse_watch_payload(byte_data: bytes) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Watch JSON payload:
      {
        "uuid": "...",
        "tsMillis": 123,
        "fsHz": 30.0,
        "acc":  [[ax,ay,az], ...],   // T x 3 (ideally 128x3)
        "gyro": [[gx,gy,gz], ...],   // T x 3 (ideally 128x3)
        "unitsAcc": "g" or "m/s^2",
        "unitsGyro": "rad/s" or "deg/s"
      }
    Returns:
      acc_128 (128,3), gyro_128 (128,3), fsHz
    """
    obj = json.loads(byte_data.decode("utf-8"))

    if isinstance(obj, list):
        raise ValueError("Payload is a JSON array; expected JSON object with keys acc/gyro.")

    fs = float(obj.get("fsHz", DEFAULT_FS_HZ))

    acc = _as_float32_array(obj.get("acc"))
    gyro = _as_float32_array(obj.get("gyro"))

    acc = _fix_nans(acc)
    gyro = _fix_nans(gyro)

    acc = _ensure_128x3(acc, "acc")
    gyro = _ensure_128x3(gyro, "gyro")

    gyro = _convert_gyro_units(gyro, obj.get("unitsGyro", "rad/s"))

    return acc, gyro, fs


def preprocess(acc: np.ndarray, gyro: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Preprocess raw IMU data to 7-channel Kalman features.

    CRITICAL: Uses trained StandardScaler (ACC_SCALER) for normalization
    to match training pipeline exactly. This is essential for preventing
    false positives caused by feature distribution mismatch.

    Args:
        acc: (T, 3) accelerometer in m/s²
        gyro: (T, 3) gyroscope in rad/s
        fs: Sampling frequency

    Returns:
        features: (T, 7) [SMV, ax, ay, az, roll, pitch, yaw]
                  - Channels 0-3 (SMV, ax, ay, az): Normalized with trained scaler
                  - Channels 4-6 (roll, pitch, yaw): Raw radians (no normalization)
    """
    T = len(acc)
    dt = 1.0 / fs

    # Compute SMV (Signal Magnitude Vector)
    smv = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))

    # Kalman filter for orientation (roll, pitch, yaw in radians)
    kf = LinearKalmanFilter()
    orientations = np.zeros((T, 3))
    for t in range(T):
        kf.predict(dt)
        kf.update(acc[t], gyro[t])
        orientations[t] = kf.get_orientation()

    # Assemble: [SMV, ax, ay, az, roll, pitch, yaw]
    features = np.hstack([smv, acc, orientations])

    # ==========================================================
    # CRITICAL FIX: Use trained scaler instead of per-window stats
    # ==========================================================
    # OLD (WRONG - causes false positives):
    #   acc_channels = features[:, :4]
    #   mean = acc_channels.mean(axis=0)      # Per-window mean
    #   std = acc_channels.std(axis=0) + 1e-8 # Per-window std
    #   features[:, :4] = (acc_channels - mean) / std
    #
    # NEW (CORRECT - matches training):
    #   Use StandardScaler fitted on entire training dataset
    # ==========================================================

    if ACC_SCALER is not None:
        # Apply trained scaler (same normalization as training)
        features[:, :4] = ACC_SCALER.transform(features[:, :4])
    else:
        # Fallback: per-window normalization (NOT recommended)
        print("[WARN] ACC_SCALER not loaded, using per-window normalization (may cause false positives)")
        acc_channels = features[:, :4]
        mean = acc_channels.mean(axis=0)
        std = acc_channels.std(axis=0) + 1e-8
        features[:, :4] = (acc_channels - mean) / std

    return features


def load_model(weights_path: str, device: str = 'cpu') -> KalmanBalancedFlexible:
    """Load model with trained weights."""
    model = KalmanBalancedFlexible(
        imu_frames=128,
        imu_channels=7,
        num_heads=4,
        num_layers=2,
        embed_dim=64,
        dropout=0.5,
        activation='relu',
        norm_first=True,
        se_reduction=4,
        acc_ratio=0.5,
        use_se=True,
        use_tap=True,
        use_pos_encoding=False
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_scaler(scaler_path: str):
    """
    Load trained StandardScaler for accelerometer normalization.

    The scaler was fitted on training data using:
    - 49 training subjects (excludes validation subjects 48, 57)
    - 3536 windows x 128 frames = 452,608 samples
    - Channels: SMV, ax, ay, az (4 channels)

    Statistics (from training):
        SMV: mean=10.87, std=6.09
        ax:  mean=-2.82, std=6.82
        ay:  mean=-2.30, std=6.26
        az:  mean=3.54, std=6.61
    """
    scaler = joblib.load(scaler_path)
    print(f"[OK] Loaded scaler: {scaler_path}")
    print(f"     mean={scaler.mean_}")
    print(f"     std={scaler.scale_}")
    return scaler


def predict(model, features: np.ndarray, device: str = 'cpu') -> dict:
    """
    Run inference on preprocessed features.

    Args:
        model: Loaded model
        features: (128, 7) preprocessed features
        device: 'cpu' or 'cuda'

    Returns:
        {'probability': float, 'is_fall': bool}
    """
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit, _ = model(x)
        prob = torch.sigmoid(logit).item()
    return {'probability': prob, 'is_fall': prob > 0.5}


# -----------------------------
# Model init
# -----------------------------
def initialize_model():
    global MODEL, ACC_SCALER

    # Load scaler first (critical for correct normalization)
    if os.path.exists(SCALER_PATH):
        ACC_SCALER = load_scaler(SCALER_PATH)
    else:
        print(f"[ERROR] SCALER_PATH not found: {SCALER_PATH}")
        print(f"        Run 'python scripts/export_scaler.py' to generate it.")
        print(f"        Falling back to per-window normalization (NOT RECOMMENDED).")
        ACC_SCALER = None

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    MODEL = load_model(MODEL_PATH, device=DEVICE)
    print(f"[OK] Loaded model weights: {MODEL_PATH}")

    # Warm-up
    with torch.no_grad():
        dummy_acc = np.zeros((128, 3), dtype=np.float32)
        dummy_gyro = np.zeros((128, 3), dtype=np.float32)
        feats = preprocess(dummy_acc, dummy_gyro, fs=DEFAULT_FS_HZ)
        _ = predict(MODEL, feats, device=DEVICE)
    print("[OK] Warm-up done.")


# -----------------------------
# NATS server
# -----------------------------
async def run():
    nc = NATS()
    initialize_model()

    print(f"Connecting to NATS: {NATS_URL}")
    await nc.connect(NATS_URL)
    print("Subscribed to 'm.*'.")

    async def handler(msg):
        try:
            t0 = time.perf_counter()

            acc, gyro, fs = parse_watch_payload(msg.data)

            # Repo pipeline: Kalman fusion -> 7ch -> model
            features = preprocess(acc, gyro, fs=fs)  # (128, 7)
            out = predict(MODEL, features, device=DEVICE)
            prob = float(out["probability"])

            dur_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[{msg.subject}] prob={prob:.4f} | latency={dur_ms:.1f} ms")

            await msg.respond(str(prob).encode("utf-8"))

        except Exception as e:
            print(f"Error processing '{msg.subject}': {e}")
            await msg.respond(b"NaN")

    await nc.subscribe("m.*", cb=handler)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await nc.drain()


if __name__ == "__main__":
    asyncio.run(run())
