from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import joblib
import io
import os

# ================= CONFIG =================
API_KEY = "test_123456"
MODEL_PATH = "voice_model.pkl"

# ================= APP =================
app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0.0"
)

# ================= LOAD MODEL =================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file voice_model.pkl not found")

model = joblib.load(MODEL_PATH)

# ================= REQUEST SCHEMA =================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ================= ROOT (GET) =================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI Voice Detection API is running. Use POST /api/voice-detection"
    }

# ================= FEATURE EXTRACTION =================
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))

    features = np.hstack([mfcc_mean, zcr, rms, pitch])
    return features.reshape(1, -1)

# ================= EXPLANATION ENGINE =================
def build_explanation(pred, confidence, zcr, pitch):
    if pred == 1:  # AI
        reasons = []
        if pitch < 140:
            reasons.append("low pitch variation")
        if zcr < 0.05:
            reasons.append("uniform speech pattern")
        if not reasons:
            reasons.append("synthetic voice characteristics")

        return f"Detected AI-generated voice due to {', '.join(reasons)}."

    else:  # HUMAN
        reasons = []
        if pitch > 160:
            reasons.append("natural pitch variation")
        if zcr > 0.07:
            reasons.append("irregular speech dynamics")
        if not reasons:
            reasons.append("organic human speech traits")

        return f"Detected human voice due to {', '.join(reasons)}."

# ================= MAIN API (POST ONLY) =================
@app.post("/api/voice-detection")
def voice_detection(
    payload: VoiceRequest,
    x_api_key: str = Header(None)
):
    # ---------- API KEY ----------
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---------- DECODE AUDIO ----------
    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio data")

    # ---------- FEATURES ----------
    features = extract_features(audio, sr)

    # ---------- PREDICTION ----------
    proba = model.predict_proba(features)[0]
    pred = int(np.argmax(proba))
    confidence = float(np.max(proba))

    # ---------- EXTRA METRICS ----------
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    pitch = float(np.mean(librosa.yin(audio, fmin=50, fmax=300)))

    classification = "AI_GENERATED" if pred == 1 else "HUMAN"

    explanation = build_explanation(pred, confidence, zcr, pitch)

    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
