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
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=330)
    plt.colorbar(label='dB')
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.savefig("my_fig.png")
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