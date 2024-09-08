import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import dagshub
import joblib

# Initialize DagsHub with MLflow
dagshub.init(repo_owner='sarthakdevil', repo_name='Model', mlflow=True)

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

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=None):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)

    train_accuracy = rf_model.score(X_train, y_train)
    val_accuracy = rf_model.score(X_val, y_val)

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Log metrics to MLflow
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)

    return rf_model

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

    assert len(features_array) == len(labels), "Mismatch between features and labels length"

    # Flatten the feature array for Random Forest input
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest model
    with mlflow.start_run():
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", None)

        rf_model = train_random_forest(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            n_estimators=100,
            max_depth=None
        )

        # Predict and calculate accuracy
        y_pred = rf_model.predict(X_test_scaled)
        test_accuracy = rf_model.score(X_test_scaled, y_test)

        # Log the test accuracy
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"Test Accuracy: {test_accuracy}")

        # Plot training vs test accuracy
        accuracies = [rf_model.score(X_train_scaled, y_train), test_accuracy]
        labels = ['Training Accuracy', 'Test Accuracy']

        plt.figure(figsize=(8, 6))
        plt.bar(labels, accuracies, color=['blue', 'green'])
        plt.ylim([0, 1])
        plt.ylabel('Accuracy')
        plt.title('Training vs Test Accuracy')
        plt.savefig("accuracy_comparison.png")
        plt.show()

        # Log the accuracy comparison graph
        mlflow.log_artifact("accuracy_comparison.png")

        # Confusion Matrix
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["nonguns", "guns"])
        disp.plot(cmap=plt.cm.Blues)

        plt.savefig("confusion_matrix.png")
        plt.show()

        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_artifact("confusion_matrix.png")
        report_str = classification_report(y_test, y_pred)
        mlflow.log_text(report_str, "classification_report.txt")

        # Save the Random Forest model
        joblib.dump(rf_model, "model_random_forest_v1.pkl")
        mlflow.log_artifact("model_random_forest_v1.pkl")

        print("Model, classification report, and accuracy comparison graph logged with MLflow and DagsHub.")
