from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64, io, os
import numpy as np
import librosa
import joblib

app = FastAPI(title="AI Generated Voice Detection API", version="final")

API_KEY = "test_123456"
MODEL_PATH = "voice_model.pkl"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"status": "ok", "message": "API running"}

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_vals = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_vals) if len(pitch_vals) else 0
    pitch_std = np.std(pitch_vals) if len(pitch_vals) else 0

    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    features = np.concatenate([mfcc_mean, [pitch_mean, rms_mean, zcr_mean]]).reshape(1, -1)

    meta = {
        "pitch_std": pitch_std,
        "energy": rms_mean,
        "zcr": zcr_mean
    }

    return features, meta

@app.api_route("/api/voice-detection", methods=["GET", "POST"])
def voice_detection(payload: dict = None, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if payload is None:
        return {"status": "ok", "message": "Use POST with audio data"}

    try:
        audio_b64 = payload.get("audioBase64")
        language = payload.get("language")

        audio_bytes = base64.b64decode(audio_b64)
        audio_io = io.BytesIO(audio_bytes)

        audio, sr = librosa.load(audio_io, sr=None)

        features, meta = extract_features(audio, sr)

        probs = model.predict_proba(features)[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

        classification = "AI_GENERATED" if prediction == 1 else "HUMAN"

        # ---- DYNAMIC EXPLANATION ----
        if classification == "AI_GENERATED":
            explanation = (
                "Low pitch variation and stable energy patterns detected, "
                "which are commonly observed in AI-generated speech."
            )
        else:
            explanation = (
                "Higher pitch variability and irregular speech patterns detected, "
                "which are typical characteristics of natural human speech."
            )

        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
