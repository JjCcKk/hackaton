import numpy as np
import librosa
import speech_recognition as sr
import soundfile as sf
import translators as ts

def compute_spect(audio):
    sf.write('my_song.wav', audio, 48000, subtype='PCM_16')
    y, sr = librosa.load("my_song.wav", sr=48000)
    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=330, n_fft=32)
    spectrogram = np.abs(mel_signal)
    
    return spectrogram

def get_the_text(audio):
    sf.write('my_song.wav', audio, 48000, subtype='PCM_16')
    r = sr.Recognizer()
    with sr.AudioFile("my_song.wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
    return text


def get_traduction(texte, langage):
    return ts.google(texte, to_language=langage)