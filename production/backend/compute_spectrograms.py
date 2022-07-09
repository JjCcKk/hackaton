import scipy.io
import numpy as np
import librosa


def compute_spect(audio):
    scipy.io.wavfile.write("my_song.wav", audio[0], np.array(audio[1]))
    y, sr = librosa.load("my_song.wav", sr=audio[0])
    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=330, n_fft=32)
    spectrogram = np.abs(mel_signal)