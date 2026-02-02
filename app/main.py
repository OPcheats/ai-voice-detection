from fastapi import FastAPI, Header, HTTPException
from typing import Optional
import base64
import librosa
import tempfile
import os

app = FastAPI()

API_KEY = "test_123456"


@app.get("/")
def root():
    return {
        "message": "API is running",
        "status": "ok"
    }


@app.post("/api/voice-detection")
def voice_detection(
    payload: dict,
    x_api_key: Optional[str] = Header(None)
):
    # =========================
    # API KEY CHECK
    # =========================
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # =========================
    # STEP-2: REQUEST VALIDATION
    # =========================
    language = payload.get("language")
    audio_format = payload.get("audioFormat")
    audio_base64 = payload.get("audioBase64")

    if not language:
        return {"status": "error", "message": "Invalid request format"}

    if audio_format != "mp3":
        return {"status": "error", "message": "Invalid request format"}

    if not audio_base64:
        return {"status": "error", "message": "Invalid request format"}

    # =========================
    # STEP-3: BASE64 → MP3 DECODE
    # =========================
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        return {"status": "error", "message": "Audio decoding failed"}

    # =========================
    # TEMPORARY MP3 FILE
    # =========================
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
    except Exception:
        return {"status": "error", "message": "Audio file creation failed"}

    # =========================
    # LOAD AUDIO USING LIBROSA
    # =========================
    try:
        # sr=None → no resampling (PDF rule)
        audio_data, sample_rate = librosa.load(temp_audio_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    except Exception:
        os.remove(temp_audio_path)
        return {"status": "error", "message": "Audio loading failed"}

    # =========================
    # CLEANUP TEMP FILE
    # =========================
    os.remove(temp_audio_path)

    # =========================
    # SUCCESS RESPONSE (DAY-2)
    # =========================
    return {
        "status": "success",
        "language": language,
        "classification": "HUMAN",
        "confidenceScore": 0.5,
        "explanation": "Audio decoded and loaded successfully"
    }
