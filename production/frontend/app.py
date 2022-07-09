import gradio as gr


def greet(audio:tuple) -> tuple:
    print("hello ma gueule")


audio = gr.inputs.Audio(source="microphone", 
                        label='First recording', 
                        optional=False)


iface = gr.Interface(fn=greet,

                    inputs=[audio], 
                    outputs=["text"],

                    title="Sarcasm detection",
                    description='''Happy hackathon !!''',

                    theme="peach",
                    )


if __name__ == "__main__":
    app, local_url, share_url = iface.launch()