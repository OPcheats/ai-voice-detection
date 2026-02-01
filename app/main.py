from fastapi import FastAPI, Header, HTTPException
from typing import Optional

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
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {
        "status": "success",
        "language": payload.get("language"),
        "classification": "HUMAN",
        "confidenceScore": 0.5,
        "explanation": "Placeholder response"
    }
