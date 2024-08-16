import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time


def towav(file_path):
    base = os.path.splitext(file_path)[0]
    audio = AudioSegment.from_mp3(file_path)
    audio.export(f"{base}.wav", format="wav")
    print(f"Converted {file_path} to {base}.wav")


def convert_all_mp3_to_wav(input_sound_dir):
    for f in os.listdir(input_sound_dir):
        if f.endswith(".mp3"):
            towav(os.path.join(input_sound_dir, f))
    wav_files = [f for f in os.listdir(input_sound_dir) if f.endswith(".wav")]
    return wav_files


def find_magnitude_peaks(freq, audio_fft, num_peaks=20, min_freq=0, max_freq=3000):
    mask = (freq >= min_freq) & (freq <= max_freq)
    freq_filtered = freq[mask]
    audio_fft_filtered = audio_fft[mask]
    peaks, _ = find_peaks(audio_fft_filtered)
    peak_frequencies = freq_filtered[peaks]
    peak_magnitudes = audio_fft_filtered[peaks]
    sorted_indices = np.argsort(peak_magnitudes)[::-1]
    top_peak_frequencies = peak_frequencies[sorted_indices][:num_peaks]
    top_peak_magnitudes = peak_magnitudes[sorted_indices][:num_peaks]
    top_peak_magnitudes = top_peak_magnitudes / np.max(top_peak_magnitudes)
    return top_peak_frequencies, top_peak_magnitudes


def read_audio(file_path):
    sample_rate, audio_data = wav.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    return sample_rate, audio_data


def compute_fourier_transform(audio_data, sample_rate):
    n = len(audio_data)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)
    audio_fft = np.fft.fft(audio_data)
    return freq, np.abs(audio_fft)


def create_cnn_model(input_shape):
    model = models.Sequential(
        [
            layers.Conv1D(32, 3, activation="relu", input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def process_sound_files(input_dir, label):
    wav_files = convert_all_mp3_to_wav(input_dir)
    features_list = []
    labels = []

    for wav_file in wav_files:
        wav_file_path = os.path.join(input_dir, wav_file)
        sample_rate, audio_data = read_audio(wav_file_path)
        freq, audio_fft = compute_fourier_transform(audio_data, sample_rate)
        peak_frequencies, peak_magnitudes = find_magnitude_peaks(freq, audio_fft)
        features = np.concatenate((peak_frequencies, peak_magnitudes))
        features_list.append(features)
        labels.append(label)

        print(f"Processed: {wav_file}")

    return np.array(features_list), np.array(labels)


if __name__ == "__main__":
    input_gun_dir = "./dataset/train/guns/"
    input_nongun_dir = "./dataset/train/non guns/"
    output_directory = "./graphs/"
    test_gun_dir = "./dataset/test/guns/"
    test_nongun_dir = "./dataset/test/non guns/"
    os.makedirs(output_directory, exist_ok=True)

    gun_features, gun_labels = process_sound_files(input_gun_dir, label=1)
    nongun_features, nongun_labels = process_sound_files(input_nongun_dir, label=0)

    features_array = np.vstack((gun_features, nongun_features))
    labels = np.hstack((gun_labels, nongun_labels))

    assert len(features_array) == len(
        labels
    ), "Mismatch between features and labels length"

    features_reshaped = features_array.reshape(
        features_array.shape[0], features_array.shape[1], 1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features_reshaped, labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    model = create_cnn_model((features_array.shape[1], 1))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=50,  # Start with 50 and use early stopping to halt if necessary
        batch_size=16,  # Start with 16; adjust if necessary based on memory usage
        validation_split=0.2,
        verbose=1,
    )

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.savefig("model_analysis.png")
    plt.show()

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy on validation set: {test_accuracy:.4f}")

    model.save("model_fft.h5")

    test_features_gun, test_labels_gun = process_sound_files(test_gun_dir, label=1)
    test_features_nongun, test_labels_nongun = process_sound_files(
        test_nongun_dir, label=0
    )

    test_features = np.vstack((test_features_gun, test_features_nongun))
    test_labels = np.hstack((test_labels_gun, test_labels_nongun))

    test_features_reshaped = test_features.reshape(
        test_features.shape[0], test_features.shape[1], 1
    )
    test_features_scaled = scaler.transform(
        test_features_reshaped.reshape(-1, test_features_reshaped.shape[-1])
    ).reshape(test_features_reshaped.shape)

    test_loss, test_accuracy = model.evaluate(
        test_features_scaled, test_labels, verbose=0
    )
    print(f"Test accuracy on new data: {test_accuracy:.4f}")
