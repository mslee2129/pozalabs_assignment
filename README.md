# pozalabs_assignment

## This repository contains code for the task of timbre combination.

#### To summarise, recreating Magenta's NSynth model was attempted but it was decided to be unfeasible due to resource limitations. CycleGAN was considered next, given the unpaired dataset, and the base implementation is provided, but was again deemed to be too complex given the timeline. The VAE was implemented as a potential approach due to ease of implementation, but it should be noted that it may not be the ideal architecture for this task.
#### VAE folder contains code for Variational Autoencoder, which can be trained by running train.py. VAE was not fully trained or tested due to time constraints.
#### GAN folder contains unfinished implementation of the GAN approach. Code taken from the tensorflow version of the CycleGAN paper.
#### preprocess.py contains code to convert wav files to spectrogram (both numpy array and mel-spectrogram image).
