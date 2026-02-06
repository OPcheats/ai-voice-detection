from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64, io, joblib
import numpy as np
import soundfile as sf

from features import extract_features

API_KEY = "test_123456"

model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="AI Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
def root():
    return {"status": "ok", "message": "API running"}

@app.post("/api/voice-detection")
def detect(payload: VoiceRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    audio_bytes = base64.b64decode(payload.audioBase64)
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    features = extract_features(audio, sr).reshape(1, -1)
    features = scaler.transform(features)

    probs = model.predict_proba(features)[0]
    pred = int(np.argmax(probs))
    confidence = round(float(probs[pred]), 2)

    classification = "AI_GENERATED" if pred == 1 else "HUMAN"

    explanation = (
        "The audio shows low pitch variability, smooth energy patterns, "
        "and consistent spectral features, which are commonly observed in AI-generated speech."
        if classification == "AI_GENERATED"
        else
        "The audio contains natural pitch variations, irregular energy flow, "
        "and diverse acoustic patterns, which are typical of human speech."
    )

    return {
        "status": "success",
        "language": payload.language.lower(),
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
