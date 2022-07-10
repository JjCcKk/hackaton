from fastapi import FastAPI
from httpx import get
from compute_spectrograms import compute_spect, get_the_text, get_traduction
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/preprocess")
def preprocess(input_user:dict):
    spec = compute_spect(np.array(input_user["Enregistrement"], dtype='int32')).tolist()
    return {"Spectrogram":spec}

@app.post("/gettext")
def get_text(input_user:dict):
    texte = get_the_text(np.array(input_user["Enregistrement"], dtype='int32'))
    traduction = get_traduction(texte, input_user["Langue"])
    return {"Texte":texte, "Traduction":traduction}