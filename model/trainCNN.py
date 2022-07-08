#from ntpath import join
import torch as tr
#import torch.cuda
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from dataset import SarcasmDataset
from cnn import MelCNN
from cnn import SelfAttention


BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 0.00002

ANNOTATIONS_FILE = "annotations.csv"
SAMPLE_RATE = 44100
NUM_SAMPLES = 44100*5 #The Model works with samples of 5s length.


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, lable in data_loader:
        input = input.to(device)
        if lable:
            lable = 1
        else:
            lable = 0

        #print("Shape of sample: " + str(input.size()))
        input = input.float()

        # calculate loss
        prediction = model(input)
        #print(prediction.size())
        prediction = tr.squeeze(prediction)
        loss = loss_fn(prediction,lable)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    #Calculate loss on test set here!

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    if tr.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    MELi = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=330, n_mels=32, normalized=True)

    usd = SarcasmDataset(ANNOTATIONS_FILE,
                            MELi, #Callable Object that was instantiated before.
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(len(usd))
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    SAi = SelfAttention(9,device).to(device) #Parameter is dim before attention-block
    SAi = SAi.float()
    MelCNNi = MelCNN(SAi).to(device)
    MelCNNi = MelCNNi.float()

    print(MelCNNi)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()
    optimiser = tr.optim.Adam(MelCNNi.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(MelCNNi, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    tr.save(MelCNNi.state_dict(), "melCNN.pth")
    print("Trained convVAE saved at melCNN.pth")
