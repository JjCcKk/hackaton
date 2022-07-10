from fastapi import FastAPI
from compute_spectrograms import get_the_text, get_traduction, compute_pred
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/prediction")
def preprocess(input_user:dict):
    return {"Prediction":compute_pred(np.array(input_user["Enregistrement"], dtype='int32'))}

@app.post("/gettext")
def get_text(input_user:dict):
    texte = get_the_text(np.array(input_user["Enregistrement"], dtype='int32'))
    traduction = get_traduction(texte, input_user["Langue"])
    return {"Texte":texte, "Traduction":traduction}