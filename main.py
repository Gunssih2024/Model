from time import time
import joblib
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
import librosa

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


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


def compute_mfcc(audio_data, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(
        y=audio_data.astype(float), sr=sample_rate, n_mfcc=n_mfcc
    )
    return mfccs


def read_audio(file_path):
    sample_rate, audio_data = wav.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    return sample_rate, audio_data


def train_with_manual_backprop(
    model, X_train, y_train, X_val, y_val, epochs, batch_size
):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    # Prepare the training and validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(
        batch_size
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Initialize metrics for each epoch
        epoch_train_loss = tf.keras.metrics.Mean()
        epoch_train_accuracy = tf.keras.metrics.BinaryAccuracy()
        epoch_val_loss = tf.keras.metrics.Mean()
        epoch_val_accuracy = tf.keras.metrics.BinaryAccuracy()

        # Training loop
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update metrics
            epoch_train_loss.update_state(loss_value)
            epoch_train_accuracy.update_state(y_batch_train, logits)

            if step % 10 == 0:
                print(f"Training loss at step {step}: {loss_value.numpy()}")

        # End of epoch: calculate validation metrics
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)

            epoch_val_loss.update_state(val_loss_value)
            epoch_val_accuracy.update_state(y_batch_val, val_logits)

        # Store the metrics for this epoch
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

    # Return the model and the history of metrics
    history = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    return model, history


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


def add_noise(audio_data, noise_type="white", noise_level=0.005):
    if noise_type == "white":
        noise = np.random.normal(0, 1, len(audio_data))
    elif noise_type == "normal":
        noise = np.random.normal(0, 1, len(audio_data)) * np.std(audio_data)
    else:
        raise ValueError("Unsupported noise type. Use 'white' or 'normal'.")

    augmented_audio = audio_data + noise_level * noise
    return augmented_audio


def process_sound_files(input_dir, label, augment_with_noise=False):
    wav_files = convert_all_mp3_to_wav(input_dir)
    features_list = []
    labels = []

    for wav_file in wav_files:
        wav_file_path = os.path.join(input_dir, wav_file)
        sample_rate, audio_data = read_audio(wav_file_path)

        # Original MFCC computation
        mfccs = compute_mfcc(audio_data, sample_rate)
        mfcc_features = np.mean(mfccs, axis=1)  # Take the mean of each MFCC coefficient
        features_list.append(mfcc_features)
        labels.append(label)

        print(f"Processed: {wav_file}")

        if augment_with_noise:
            # Augment with white noise
            augmented_audio_white = add_noise(audio_data, noise_type="white")
            mfccs_white = compute_mfcc(augmented_audio_white, sample_rate)
            mfcc_features_white = np.mean(mfccs_white, axis=1)
            features_list.append(mfcc_features_white)
            labels.append(label)
            print(f"Processed with white noise: {wav_file}")

            # Augment with normal noise
            augmented_audio_normal = add_noise(audio_data, noise_type="normal")
            mfccs_normal = compute_mfcc(augmented_audio_normal, sample_rate)
            mfcc_features_normal = np.mean(mfccs_normal, axis=1)
            features_list.append(mfcc_features_normal)
            labels.append(label)
            print(f"Processed with normal noise: {wav_file}")

    return np.array(features_list), np.array(labels)


if __name__ == "__main__":
    current_dir = os.getcwd()
    input_gun_dir = os.path.join(current_dir, "dataset/train/guns/")
    input_nongun_dir = os.path.join(current_dir, "dataset/train/non guns/")
    output_directory = os.path.join(current_dir, "graph/")
    test_gun_dir = os.path.join(current_dir, "dataset/test/guns/")
    test_nongun_dir = os.path.join(current_dir, "dataset/test/non guns/")

    gun_features, gun_labels = process_sound_files(
        input_gun_dir, label=1, augment_with_noise=True
    )
    nongun_features, nongun_labels = process_sound_files(
        input_nongun_dir, label=0, augment_with_noise=True
    )

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
    model, history = train_with_manual_backprop(
        model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=50, batch_size=16
    )

    joblib.dump(scaler, "scaler.joblib")
    plt.figure(figsize=(10, 6))

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.plot(
        epochs_range, history["train_accuracy"], label="Train Accuracy", color="blue"
    )
    plt.plot(
        epochs_range,
        history["val_accuracy"],
        label="Validation Accuracy",
        color="green",
    )
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="red")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="orange")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.title("Accuracy and Loss over Epochs")
    plt.legend(loc="best")
    plt.grid(True)

    plot_path = os.path.join(output_directory, "accuracy_loss_graph_v2.png")
    plt.savefig(plot_path)
    plt.show()
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy on validation set: {test_accuracy:.4f}")

    path = os.path.join(current_dir, "models/handrecognition_model.h5")
    model.save(path)

    test_features_gun, test_labels_gun = process_sound_files(
        test_gun_dir, label=1, augment_with_noise=True
    )
    test_features_nongun, test_labels_nongun = process_sound_files(
        test_nongun_dir, label=0, augment_with_noise=True
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
