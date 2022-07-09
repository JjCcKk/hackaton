from fastapi import FastAPI
from compute_spectrograms import compute_spect
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/preprocess")
def preprocess(input_user:dict):
    return {"Spectrogram":compute_spect(np.array(input_user["Enregistrement"])).tolist()}