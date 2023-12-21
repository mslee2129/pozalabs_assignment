import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def wav_to_mel(input_wav, output_folder, duration=5, sr=22050, hop_length=512, n_mels=128):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the audio file
    y, sr = librosa.load(input_wav, sr=sr)

    # Calculate the number of samples per snippet
    samples_per_snippet = int(duration * sr)

    # Calculate the number of snippets
    num_snippets = len(y) // samples_per_snippet

    for i in range(num_snippets):
        # Extract a 5-second snippet
        snippet = y[i * samples_per_snippet : (i + 1) * samples_per_snippet]

        # Convert the snippet to a mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=snippet, sr=sr, hop_length=hop_length, n_mels=n_mels)

        # Convert to decibels (log scale)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Plot and save the mel-spectrogram
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length)
        plt.tight_layout()

        # Save the mel-spectrogram as an image
        output_path = os.path.join(output_folder, input_wav[-8:-4] + f'_{i+1}.png')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def wav_to_np(input_wav, output_folder, duration=5, sr=22050, hop_length=512, n_mels=128):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the number of samples per snippet
    samples_per_snippet = int(duration * sr)

    # Load the audio file
    y, sr = librosa.load(input_wav, sr=sr)

    # Calculate the number of snippets
    num_snippets = len(y) // samples_per_snippet

    for i in range(num_snippets):
        # Extract a 5-second snippet
        snippet = y[i * samples_per_snippet : (i + 1) * samples_per_snippet]
    
        stft = librosa.stft(y=snippet,
                            hop_length=hop_length,
                            n_fft=510)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        output_path = os.path.join(output_folder, input_wav[-8:-4] + f'_{i+1}')
        np.save(output_path, log_spectrogram)

if __name__ == "__main__":
    # Specify input .wav file and output folder
    input_wav_file = 'data/train/wav/'
    output_folder = 'data/train/np/'

    # Set parameters
    duration = 5  # 5 seconds
    sampling_rate = 22050  # in Hz
    hop_length = 1724
    num_mel_filters = 128

    # Perform division and mel-spectrogram conversion
    for file in os.listdir(input_wav_file):
        if file.endswith('.wav'):
            wav_to_np(input_wav_file+file, output_folder, duration, sampling_rate, hop_length, num_mel_filters)
