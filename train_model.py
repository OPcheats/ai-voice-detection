from fastapi import FastAPI, Header, HTTPException
from typing import Optional
import base64
import librosa
import numpy as np
import tempfile
import os

app = FastAPI(
    title="AI Voice Detection API",
    version="1.0.0"
)

API_KEY = "test_123456"


@app.get("/")
def root():
    return {
        "message": "API is running",
        "status": "ok"
    }


# =========================
# FEATURE EXTRACTION
# =========================
def extract_audio_features(audio_data, sample_rate):
    features = []

    # MFCC (13 coefficients - mean)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))

    # Pitch
    pitches, _ = librosa.piptrack(y=audio_data, sr=sample_rate)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    features.append(pitch_mean)

    # Energy (RMS)
    rms = librosa.feature.rms(y=audio_data)
    features.append(np.mean(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features.append(np.mean(zcr))

    return np.array(features)


# =========================
# MAIN API ENDPOINT
# =========================
@app.post("/api/voice-detection")
def voice_detection(
    payload: dict,
    x_api_key: Optional[str] = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # =========================
    # REQUEST VALIDATION
    # =========================
    language = payload.get("language")
    audio_format = payload.get("audioFormat")
    audio_base64 = payload.get("audioBase64")

    if not language or audio_format != "mp3" or not audio_base64:
        return {
            "status": "error",
            "message": "Invalid request format"
        }

    # =========================
    # BASE64 DECODE
    # =========================
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        return {
            "status": "error",
            "message": "Audio decoding failed"
        }

    # =========================
    # TEMP MP3 FILE
    # =========================
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name
    except Exception:
        return {
            "status": "error",
            "message": "Audio file creation failed"
        }

    # =========================
    # LOAD AUDIO
    # =========================
    try:
        audio_data, sample_rate = librosa.load(temp_path, sr=None)
    except Exception:
        os.remove(temp_path)
        return {
            "status": "error",
            "message": "Audio loading failed"
        }

    # Remove temp file
    os.remove(temp_path)

    # =========================
    # DAY-3 STEP-1: FEATURES
    # =========================
    features = extract_audio_features(audio_data, sample_rate)

    # =========================
    # TEMP RESPONSE (NO ML YET)
    # =========================
    return {
        "status": "success",
        "language": language,
        "classification": "HUMAN",
        "confidenceScore": 0.5,
        "explanation": "Audio features extracted successfully"
    }
