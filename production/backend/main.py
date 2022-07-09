from fastapi import FastAPI
from compute_spectrograms import compute_spect

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/preprocess")
def preprocess():
    return {"Hello world"}