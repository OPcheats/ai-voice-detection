import os
import librosa
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATASET_DIR = "dataset"
AI_DIR = os.path.join(DATASET_DIR, "ai")
HUMAN_DIR = os.path.join(DATASET_DIR, "human")

X = []
y = []

def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, sr=None)

    # MFCC (13)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Pitch
    pitches, _ = librosa.piptrack(y=y_audio, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # RMS Energy
    rms = librosa.feature.rms(y=y_audio)
    rms_mean = np.mean(rms)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y_audio)
    zcr_mean = np.mean(zcr)

    return np.concatenate([mfcc_mean, [pitch_mean, rms_mean, zcr_mean]])

# ---- LOAD DATA ----
for file in os.listdir(AI_DIR):
    if file.endswith(".mp3") or file.endswith(".wav"):
        X.append(extract_features(os.path.join(AI_DIR, file)))
        y.append(1)  # AI

for file in os.listdir(HUMAN_DIR):
    if file.endswith(".mp3") or file.endswith(".wav"):
        X.append(extract_features(os.path.join(HUMAN_DIR, file)))
        y.append(0)  # Human

X = np.array(X)
y = np.array(y)

# ---- MODEL PIPELINE ----
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X, y)

joblib.dump(model, "voice_model.pkl")

print("✅ Model trained and saved as voice_model.pkl")
print("Features:", X.shape[1])
