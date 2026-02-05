import base64
import io
import os
import tempfile
import numpy as np
import librosa
import joblib

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ================= CONFIG =================
API_KEY = "test_123456"
MODEL_PATH = "voice_model.pkl"
MAX_AUDIO_SECONDS = 12
SAMPLE_RATE = 16000

# ================= APP =================
app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0"
)

model = joblib.load(MODEL_PATH)

# ================= REQUEST SCHEMA =================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ================= FEATURE EXTRACTION =================
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    pitch = librosa.yin(y, fmin=50, fmax=300)

    features = np.array([
        mfcc.mean(),
        mfcc.std(),
        np.mean(pitch),
        np.std(pitch),
        np.mean(rms),
        np.mean(zcr),
        np.std(zcr)
    ])

    meta = {
        "pitch_var": np.std(pitch),
        "zcr_mean": np.mean(zcr),
        "energy": np.mean(rms)
    }

    return features.reshape(1, -1), meta

# ================= EXPLANATION ENGINE =================
def build_explanation(pred, meta):
    reasons = []

    if meta["pitch_var"] > 20:
        reasons.append("higher pitch variability")
    else:
        reasons.append("stable pitch pattern")

    if meta["zcr_mean"] > 0.08:
        reasons.append("irregular speech transitions")
    else:
        reasons.append("smooth speech transitions")

    if meta["energy"] < 0.02:
        reasons.append("synthetic energy envelope")
    else:
        reasons.append("natural energy variation")

    if pred == "AI_GENERATED":
        return "Detected AI-generated voice due to " + ", ".join(reasons) + "."
    else:
        return "Detected human voice due to " + ", ".join(reasons) + "."

# ================= API =================
@app.post("/api/voice-detection")
def voice_detection(
    payload: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    try:
        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_AUDIO_SECONDS
        )

        X, meta = extract_features(y, sr)

        probs = model.predict_proba(X)[0]
        ai_prob = float(probs[1])
        human_prob = float(probs[0])

        if ai_prob > human_prob:
            classification = "AI_GENERATED"
            confidence = round(ai_prob, 2)
        else:
            classification = "HUMAN"
            confidence = round(human_prob, 2)

        explanation = build_explanation(classification, meta)

        return {
            "status": "success",
            "language": payload.language.lower(),
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.remove(audio_path)

# ================= ROOT =================
@app.get("/")
def root():
    return {"message": "AI Voice Detection API is running"}
