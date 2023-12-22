import os

import numpy as np

from model import VAE


LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 50

data_path = 'data/train/np/'

def load_data(path): # load data
    x_train = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train

if __name__ == "__main__": # run to train
    x_train = load_data(data_path)
    vae = VAE(
        input_shape=(1024, 128, 1), # input shape decreased for faster processing
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(5, 5, 3, 3, 3),
        conv_strides=((4, 4), (4, 2), (2, 2), (2, 2), (2, 1)),
        latent_space_dim=128
    )
    vae.summary()
    vae.compile(LEARNING_RATE)
    vae.train(x_train, BATCH_SIZE, EPOCHS)
    vae.save('model')