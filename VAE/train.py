import os

import numpy as np

from model import VAE


LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

data_path = 'data/train/np/'

def load_data(path):
    x_train = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train

if __name__ == "__main__":
    x_train = load_data(data_path)
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    vae.summary()
    vae.compile(LEARNING_RATE)
    vae.train(x_train, BATCH_SIZE, EPOCHS)
    vae.save('model')