import gradio as gr
import requests
import os
import ast

HF_TOKEN = os.getenv('HF_TOKEN')
hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "Rick-bot-flags")

def greet(audio:tuple) -> str:
    json_original = {
        "Enregistrement": audio[1].tolist()
    }

    res_preprocess = requests.post("http://127.0.0.1:8000/preprocess", json=json_original)
    print(ast.literal_eval(res_preprocess.text)["Spectrogram"])

    return "Hello"


audio = gr.inputs.Audio(source="microphone", label='What do you want to say ?')

title = "Sarcasm detector"
description = """
<p>
    <center>
        It is not always easy to know whether or not someone is being sarcastic. That's why we wanted to complete the existing NLP models with a sound analysis.
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlCS7X6KT6GDfkvjY5U2yi_mkqQnQbH25Eew&usqp=CAU" alt="rick" width="400"/>
    </center>
</p>
"""
article = "<p style='text-align: center'><a href='https://github.com/JjCcKk/hackaton' target='_blank'>Visit the code !!</p>"


iface = gr.Interface(fn=greet,
                    inputs=[audio], 
                    outputs=["text"],
                    title = title, 
                    flagging_callback = hf_writer, 
                    description = description, 
                    article = article)


if __name__ == "__main__":
    app, local_url, share_url = iface.launch()