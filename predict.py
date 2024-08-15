import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read audio file and return sample rate and audio data
def read_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    if len(audio_data.shape) > 1:  # If stereo, convert to mono
        audio_data = np.mean(audio_data, axis=1)
    return sample_rate, audio_data

# Function to compute MFCC features
def compute_mfcc(audio_data, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs

# Function to preprocess the audio file
def preprocess_file(file_path, scaler):
    sample_rate, audio_data = read_audio(file_path)
    mfccs = compute_mfcc(audio_data, sample_rate)
    mfcc_features = np.mean(mfccs, axis=1)  # Take the mean of each MFCC coefficient
    features_array = np.array([mfcc_features])
    
    # Reshape for the model and scale the features
    features_reshaped = features_array.reshape(features_array.shape[0], features_array.shape[1], 1)
    features_scaled = scaler.transform(features_reshaped.reshape(-1, features_reshaped.shape[-1])).reshape(features_reshaped.shape)
    
    return features_scaled

# Function to load the model and scaler, and make a prediction
def predict(file_path, model, scaler):
    preprocessed_data = preprocess_file(file_path, scaler)
    prediction = model.predict(preprocessed_data)
    return (prediction > 0.5).astype(int)[0][0]

# Load the model and scaler
model_path = "models/handrecognition_model.h5"
scaler_path = "scaler.joblib"

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Paths to test data folders
gunshot_folder = "dataset/test/gun_shot"
non_gunshot_folder = "dataset/test/non_gun"

# Collect file paths and labels
file_paths = []
true_labels = []

for file_name in os.listdir(gunshot_folder):
    file_paths.append(os.path.join(gunshot_folder, file_name))
    true_labels.append(1)  # Gunshot

for file_name in os.listdir(non_gunshot_folder):
    file_paths.append(os.path.join(non_gunshot_folder, file_name))
    true_labels.append(0)  # Non-gunshot

# Make predictions
predictions = [predict(file_path, model, scaler) for file_path in file_paths]

# Compute confusion matrix
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(true_labels, predictions, target_names=['Non-Gunshot', 'Gunshot'])
print("\nClassification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Gunshot', 'Gunshot'], yticklabels=['Non-Gunshot', 'Gunshot'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
