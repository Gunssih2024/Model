import numpy as np
import librosa
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import find_peaks
from main import (
    compute_fourier_transform,
    compute_mfcc,
    find_magnitude_peaks,
    convert_all_mp3_to_wav,
    towav,
)
import joblib


def predict_audio(model_path, audio_file_path, scaler):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Read and process the audio file
    sample_rate, audio_data = wavfile.read(audio_file_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Compute FFT
    freq, audio_fft = compute_fourier_transform(audio_data, sample_rate)
    peak_frequencies, peak_magnitudes = find_magnitude_peaks(freq, audio_fft)

    # Compute MFCC
    mfccs = compute_mfcc(audio_data, sample_rate)

    # Combine features
    fft_features = np.concatenate((peak_frequencies, peak_magnitudes))
    mfcc_features = np.mean(mfccs, axis=1)
    features = np.concatenate((fft_features, mfcc_features))

    # Reshape and scale the features
    features_reshaped = features.reshape(1, -1, 1)
    features_scaled = scaler.transform(
        features_reshaped.reshape(-1, features_reshaped.shape[-1])
    ).reshape(features_reshaped.shape)

    # Make prediction
    prediction = model.predict(features_scaled)

    # Interpret the result
    if prediction[0][0] > 0.6:
        return "Gun sound detected"
    else:
        return "Non-gun sound detected"


scaler = joblib.load("scaler.joblib")
model_path = "handrecognition_model.h5"
audio_file_path = "./test/mixkit-game-gun-shot-1662.mp3"
result = predict_audio(model_path, audio_file_path, scaler)
print(result)
