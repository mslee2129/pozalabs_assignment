import librosa

class MelToAudio:
    
    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length

    def generate(self, spectrograms):
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms):
        signals = []
        for spectrogram in zip(spectrograms):
            # reshape the log spectrogram
            log_spectrogram = spectrogram[:, :, 0]
            # log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(log_spectrogram)
            # apply Griffin-Lim
            signal = librosa.istft(spec, hop_length=self.hop_length)
            # append signal to "signals"
            signals.append(signal)
        return signals