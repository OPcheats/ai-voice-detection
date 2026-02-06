import numpy as np
import librosa

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0

    energy = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    return np.hstack([mfcc_mean, pitch, energy, zcr])  # 16
