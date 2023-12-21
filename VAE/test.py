import os
import pickle

import numpy as np
import soundfile as sf

from model import VAE
from utils import MelToAudio

PATH = 'data/train/np/'
HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"


def load_data(path):
    x_test = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_test.append(spectrogram)
            file_paths.append(file_path)
    x_test = np.array(x_test)
    x_test = x_test[..., np.newaxis]
    return x_test, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    print(file_paths)
    return sampled_spectrogrmas


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    mel_to_audio = MelToAudio(vae, HOP_LENGTH)

    specs, file_paths = load_data(PATH)

    # sample spectrograms + min max values
    sampled_specs = select_spectrograms(specs,
                                        file_paths,
                                        5)

    # generate audio for sampled spectrograms
    signals, _ = mel_to_audio.generate(sampled_specs)

    # convert spectrogram samples to audio
    original_signals = mel_to_audio.convert_spectrograms_to_audio(
        sampled_specs)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)