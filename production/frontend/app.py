import gradio as gr
import numpy as np
import librosa
import librosa.display
import scipy.io
import requests
import os

HF_TOKEN = os.getenv('HF_TOKEN')
hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "Rick-bot-flags")

def greet(audio:tuple) -> str:
    scipy.io.wavfile.write("my_song.wav", audio[0], np.array(audio[1]))
    y, sr = librosa.load("my_song.wav", sr=audio[0])
    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=330, n_fft=32)
    spectrogram = np.abs(mel_signal)

    res_preprocess = requests.get("https://j7zmal.deta.dev/preprocess")
    return res_preprocess.text


audio = gr.inputs.Audio(source="microphone", label='Record a sound')


title = "Sarcasm detector"
description = """
<p>
<center>

<img src="https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot/resolve/main/img/rick.png" alt="rick" width="200"/>
</center>
</p>
"""
article = "Hello ma gueule"


iface = gr.Interface(fn=greet,
                    inputs=[audio], 
                    outputs=["text"],
                    title = title, 
                    flagging_callback = hf_writer, 
                    description = description, 
                    article = article)


if __name__ == "__main__":
    
    app, local_url, share_url = iface.launch()