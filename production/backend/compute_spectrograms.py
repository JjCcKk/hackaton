import scipy.io
import numpy as np
import librosa

def compute_spect(audio):
    scipy.io.wavfile.write("my_song.wav", 22050, audio)
    y, sr = librosa.load("my_song.wav", sr=22050)
    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=330, n_fft=32)
    spectrogram = np.abs(mel_signal)
    
    return spectrogram