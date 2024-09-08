from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
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

def train_random_forest(X_train, y_train, X_val, y_val):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    train_accuracy = best_model.score(X_train, y_train)
    val_accuracy = best_model.score(X_val, y_val)

    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Validation Accuracy: {val_accuracy}")

    return best_model

if __name__ == "__main__":
    input_gun_dir = "./dataset/train/guns/"
    input_nongun_dir = "./dataset/train/non guns/"
    output_directory = "./graphs/"
    os.makedirs(output_directory, exist_ok=True)

    gun_features, gun_labels = process_sound_files(input_gun_dir, label=1)
    nongun_features, nongun_labels = process_sound_files(input_nongun_dir, label=0)

    features_array = np.vstack((gun_features, nongun_features))
    labels = np.hstack((gun_labels, nongun_labels))

    X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        mlflow.log_param("model_type", "Random Forest (GridSearchCV)")

        rf_model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

        y_pred = rf_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"Test Accuracy: {test_accuracy}")

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["nonguns", "guns"])
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        plt.show()

        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_artifact("confusion_matrix.png")

        plt.figure(figsize=(6, 4))
        plt.bar(['Train Accuracy', 'Test Accuracy'], [rf_model.score(X_train_scaled, y_train), test_accuracy])
        plt.title("Training vs Test Accuracy")
        plt.ylabel("Accuracy")
        plt.grid(True, axis='y')
        plt.savefig("train_vs_test_accuracy.png")
        mlflow.log_artifact("train_vs_test_accuracy.png")
        plt.show()

        joblib.dump(rf_model, "model_random_forest_v1.pkl")
        mlflow.log_artifact("model_random_forest_v1.pkl")

        print("Model, accuracy graph, and classification report logged with MLflow and DagsHub.")
