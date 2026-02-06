import os
import librosa
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features import extract_features

DATASET = "dataset"
X, y = [], []

def load(folder, label):
    for f in os.listdir(folder):
        if f.endswith(".wav") or f.endswith(".mp3"):
            audio, sr = librosa.load(
                os.path.join(folder, f),
                sr=16000,
                mono=True,
                duration=10
            )
            X.append(extract_features(audio, sr))
            y.append(label)

load(os.path.join(DATASET, "human"), 0)
load(os.path.join(DATASET, "ai"), 1)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)  # MUST be (?,16)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

joblib.dump(model, "voice_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Training done")
