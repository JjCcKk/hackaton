import gradio as gr
import numpy as np
import librosa
import librosa.display
import scipy.io
import matplotlib.pyplot as plt


def greet(audio:tuple) -> str:
    scipy.io.wavfile.write("my_song.wav", audio[0], np.array(audio[1]))
    y, sr = librosa.load("my_song.wav", sr=audio[0])
    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=330, n_fft=32)
    spectrogram = np.abs(mel_signal)
    print(spectrogram)
    return "hello ma gueule"


audio = gr.inputs.Audio(source="microphone", label='Record a sound')


iface = gr.Interface(fn=greet,
                    inputs=[audio], 
                    outputs=["text"],
                    title="Sarcasm detection",
                    description='''Happy hackathon !!''',
                    theme="peach",
                    )


if __name__ == "__main__":
    app, local_url, share_url = iface.launch()