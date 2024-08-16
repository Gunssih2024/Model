import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from scipy.signal import find_peaks

# Define necessary functions (e.g., read_audio, compute_fourier_transform, find_magnitude_peaks)


def read_audio(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    return sample_rate, audio_data


def compute_fourier_transform(audio_data, sample_rate):
    n = len(audio_data)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)
    audio_fft = np.abs(np.fft.fft(audio_data))
    return freq, audio_fft


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


def process_sound_files(input_dir, label):
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
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
    # Load your saved model
    model = tf.keras.models.load_model("model_fft_v2.h5")

    # Load test data
    test_gun_dir = "./dataset/test/guns/"
    test_nongun_dir = "./dataset/test/non guns/"

    test_features_gun, test_labels_gun = process_sound_files(test_gun_dir, label=1)
    test_features_nongun, test_labels_nongun = process_sound_files(
        test_nongun_dir, label=0
    )

    test_features = np.vstack((test_features_gun, test_features_nongun))
    test_labels = np.hstack((test_labels_gun, test_labels_nongun))

    # Reshape and scale the test features
    test_features_reshaped = test_features.reshape(
        test_features.shape[0], test_features.shape[1], 1
    )
    scaler = StandardScaler()
    test_features_scaled = scaler.fit_transform(
        test_features_reshaped.reshape(-1, test_features_reshaped.shape[-1])
    ).reshape(test_features_reshaped.shape)

    # Generate predictions
    y_pred = model.predict(test_features_scaled)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    # Create and save the confusion matrix
    cm = confusion_matrix(test_labels, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Set")
    plt.savefig("confusion_matrix.png")  # Save the confusion matrix as an image
    plt.show()

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(
        test_features_scaled, test_labels, verbose=0
    )
    print(f"Test accuracy on new data: {test_accuracy:.4f}")
