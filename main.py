import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa


def extract_features(file_path, n_mels=128, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec


def read_audio(file_path):
    sample_rate, audio_data = wav.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    print("sample_rate\n", sample_rate)
    print("audio data\n", audio_data)
    return sample_rate, audio_data


def compute_fourier_transform(audio_data, sample_rate):
    n = len(audio_data)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)
    print(freq)
    audio_fft = np.fft.fft(audio_data)
    print("audio fft \n", audio_fft)
    print(len(audio_fft))
    return freq, np.abs(audio_fft)


def save_combined_fourier_transform_plot(
    freq_list, audio_fft_list, file_names, output_file
):
    plt.figure(figsize=(12, 6))
    for freq, audio_fft, file_name in zip(freq_list, audio_fft_list, file_names):
        plt.plot(freq, audio_fft, label=os.path.splitext(file_name)[0])
    plt.title("Combined Fourier Transform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 3000)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def plot_frequency_histogram(freq, audio_fft, file_name, freq_range, output_file):
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq_filtered = freq[mask]
    fft_filtered = audio_fft[mask]

    plt.figure(figsize=(12, 6))
    plt.hist(freq_filtered, bins=100, weights=fft_filtered, edgecolor="black")
    plt.title(f"Frequency Distribution for {os.path.splitext(file_name)[0]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude Count")
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    input_directory = "./guns/"
    output_directory = "./graphs/"
    os.makedirs(output_directory, exist_ok=True)

    wav_files = [f for f in os.listdir(input_directory) if f.endswith(".wav")]

    freq_list = []
    audio_fft_list = []
    file_names = []

    for wav_file in wav_files:
        wav_file_path = os.path.join(input_directory, wav_file)

        sample_rate, audio_data = read_audio(wav_file_path)
        freq, audio_fft = compute_fourier_transform(audio_data, sample_rate)

        features = extract_features(wav_file_path)
        print("features\n", len(features))

        freq_list.append(freq)
        audio_fft_list.append(audio_fft)
        file_names.append(wav_file)

        print(f"Processed: {wav_file}")

        output_histogram_file = os.path.join(
            output_directory, f"{os.path.splitext(wav_file)[0]}_histogram.png"
        )
        plot_frequency_histogram(
            freq, audio_fft, wav_file, (0, 3000), output_histogram_file
        )

    output_file = "combined_fourier_transform.png"
    output_file_path = os.path.join(output_directory, output_file)
    save_combined_fourier_transform_plot(
        freq_list, audio_fft_list, file_names, output_file_path
    )
