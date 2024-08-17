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
from sklearn.metrics import classification_report
import mlflow
import dagshub
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

def train_with_manual_backprop(
    model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate=0.0001
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
        batch_size
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        epoch_train_loss = tf.keras.metrics.Mean()
        epoch_train_accuracy = tf.keras.metrics.BinaryAccuracy()
        epoch_val_loss = tf.keras.metrics.Mean()
        epoch_val_accuracy = tf.keras.metrics.BinaryAccuracy()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            epoch_train_loss.update_state(loss_value)
            epoch_train_accuracy.update_state(y_batch_train, logits)

            if step % 10 == 0:
                print(f"Training loss at step {step}: {loss_value.numpy()}")

        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)

            epoch_val_loss.update_state(val_loss_value)
            epoch_val_accuracy.update_state(y_batch_val, val_logits)

        train_loss.append(epoch_train_loss.result().numpy())
        train_accuracy.append(epoch_train_accuracy.result().numpy())
        val_loss.append(epoch_val_loss.result().numpy())
        val_accuracy.append(epoch_val_accuracy.result().numpy())

        print(
            f"Epoch {epoch+1} - Training Loss: {epoch_train_loss.result().numpy()}, "
            f"Training Accuracy: {epoch_train_accuracy.result().numpy()}, "
            f"Validation Loss: {epoch_val_loss.result().numpy()}, "
            f"Validation Accuracy: {epoch_val_accuracy.result().numpy()}"
        )

    history = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    return model, history

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

    assert len(features_array) == len(labels), "Mismatch between features and labels length"

    features_reshaped = features_array.reshape(features_array.shape[0], features_array.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(features_reshaped, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    model = create_cnn_model((features_array.shape[1], 1))

    with mlflow.start_run():
        mlflow.log_param("model_type", "CNN with FFT Features")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param('parameter name', 'value')
        mlflow.log_metric('metric name', 1)

        model, history = train_with_manual_backprop(
            model,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            epochs=50,
            batch_size=16,
        )

        plt.figure(figsize=(12, 6))
        plt.plot(history["train_accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.plot(history["train_loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training & Validation Accuracy and Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.legend()
        plt.savefig("model_analysis_combined.png")
        plt.show()

        # Log the plot as an artifact
        mlflow.log_artifact("model_analysis_combined.png")

        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"Test accuracy on validation set: {test_accuracy:.4f}")

        # Predict on test data
        y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32").flatten()
        y_pred_classes = np.round(y_pred).astype(int)
        # Generate the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)

        # Save the figure
        plt.savefig("confusion_matrix.png")
        plt.show()

        # Log the classification report as a JSON file
        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_artifact("confusion_matrix.png")
        # Alternatively, log as a text file
        report_str = classification_report(y_test, y_pred)
        mlflow.log_text(report_str, "classification_report.txt")

        # Save the model
        model.save("model_fft_v2.h5")
        mlflow.keras.log_model(model, "model")

        print("Model and classification report logged with MLflow and DagsHub.")