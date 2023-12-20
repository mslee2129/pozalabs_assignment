import librosa
import os

# Load and preprocess the data
def load_data(data_path):
    files = os.listdir(data_path)
    data = []
    for file in files:
        if file.endswith(".wav"):
            audio, _ = librosa.load(os.path.join(data_path, file), sr=None)
            data.append(audio)
    return data


def wav_to_mel():
    pass

def mel_to_wav():
    pass