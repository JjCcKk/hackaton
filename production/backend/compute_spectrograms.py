from predict import getPrediction
import speech_recognition as sr
import soundfile as sf
import translators as ts

def compute_pred(audio):
    sf.write('my_song.wav', audio, 48000, subtype='PCM_16')
    if getPrediction("my_song.wav").tolist()[0][0][0] > 0.5:
        return "Sarcastic"
    return "Non Sarcastic"

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