import numpy as np
import tensorflow as tf
from audio_processing import (
    towav,
    read_audio,
    compute_fourier_transform,
    find_magnitude_peaks,
)
import joblib


def predict_audio(model_path, audio_file_path, scaler):
    if audio_file_path.endswith(".mp3"):
        audio_file_path = towav(audio_file_path)

    model = tf.keras.models.load_model(model_path)

    sample_rate, audio_data = read_audio(audio_file_path)

    freq, audio_fft = compute_fourier_transform(audio_data, sample_rate)
    peak_frequencies, peak_magnitudes = find_magnitude_peaks(freq, audio_fft)

    # Optionally add more feature extraction like MFCC or others

    fft_features = np.concatenate((peak_frequencies, peak_magnitudes))
    features_reshaped = fft_features.reshape(1, -1, 1)
    features_scaled = scaler.transform(
        features_reshaped.reshape(-1, features_reshaped.shape[-1])
    ).reshape(features_reshaped.shape)

    prediction = model.predict(features_scaled)
    if prediction[0][0] > 0.6:
        return "Gun sound detected"
    else:
        return "Non-gun sound detected"


# Example usage
scaler = joblib.load("./models/scaler.joblib")
model_path = "./models/model_fft.h5"
audio_file_path = "./test/balloon-pop.mp3"
result = predict_audio(model_path, audio_file_path, scaler)
print(result)
