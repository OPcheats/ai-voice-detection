AI-Generated Voice Detection (Problem 1)
Project Purpose

This project is created for India AI Impact Buildathon 2026
Problem Statement 1: AI-Generated Voice Detection

The goal is to build a secure REST API that accepts a Base64-encoded MP3 audio and classifies the voice as:

AI_GENERATED

HUMAN

⚠️ Day-1 scope:
Only API setup, authentication, and request/response validation are completed.
AI logic will be implemented from Day-2 onward.

2️⃣ What Is Completed on Day-1

Backend API is running successfully

Required endpoint is created

API Key authentication is working

Request & response format matches problem statement

API tested successfully using FastAPI (Swagger UI)

3️⃣ What Is NOT Included in Day-1

No audio processing

No AI / ML model

No Base64 decoding

No database

No deployment

This is intentional and correct for Day-1.

4️⃣ How to Access the API
Base URL (Local)
http://127.0.0.1:8000

Swagger UI (Recommended for Testing)
http://127.0.0.1:8000/docs

5️⃣ API Endpoints
✅ Root Endpoint

Used only to confirm the API is running.

Method: GET

Path: /

Expected response indicates API is active.

✅ Voice Detection Endpoint (Main)

Method: POST

Path: /api/voice-detection

This endpoint will be used by the evaluation system.

6️⃣ Authentication (Very Important)

All requests must include an API key in the header.

Header name:

x-api-key


Current API Key (for testing):

test_123456


Requests without this key will fail.

7️⃣ Request Format (Strict)
Headers
Content-Type: application/json
x-api-key: test_123456

Body (JSON)
{
  "language": "Hindi",
  "audioFormat": "mp3",
  "audioBase64": "ANY_TEST_STRING"
}


Note: For Day-1 testing, the audioBase64 value can be any dummy text.

8️⃣ Response Format (Day-1 Placeholder)
{
  "status": "success",
  "language": "Hindi",
  "classification": "HUMAN",
  "confidenceScore": 0.5,
  "explanation": "Placeholder response"
}


This response confirms the API is working correctly.

9️⃣ How Teammates Should Test (Step-by-Step)
Method 1: FastAPI Swagger UI (Recommended)

Open browser

Go to:

http://127.0.0.1:8000/docs


Select POST /api/voice-detection

Click Try it out

Add header:

x-api-key: test_123456


Paste request JSON

Click Execute

Check response

If response appears → API is working.

