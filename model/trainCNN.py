#from ntpath import join
import torch as tr
#import torch.cuda
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import torch.nn as nn
from dataset import SarcasmDataset
from cnn import MelCNN
from cnn import SelfAttention
from configtrain import TRAIN_FILE, TEST_FILE

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.00002

TRAIN_FILE = TRAIN_FILE
TEST_FILE = TEST_FILE
SAMPLE_RATE = 44100
NUM_SAMPLES = 44100*5 #The Model works with samples of 5s length.


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, train_data_loader, loss_fn, optimiser, device, test_data_loader):
    for input, lable in train_data_loader:
        input = input.to(device)
        lable = lable.bool().int().float().to(device)

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
    testLoss = []
    for i,lable in test_data_loader:
        i = i.to(device)
        lable = lable.bool().int().float().to(device)
        i = i.float()
        # calculate loss
        prediction = model(i)
        prediction = tr.squeeze(prediction)
        loss = loss_fn(prediction,lable)
        testLoss.append(loss.item())
    print(f"test_loss: {np.mean(testLoss)}")

def train(model, train_data_loader, loss_fn, optimiser, device, epochs, test_data_loader):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, train_data_loader, loss_fn, optimiser, device, test_data_loader)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    if tr.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    MELi = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=330, n_mels=32, normalized=True)

    usdTrain = SarcasmDataset(TRAIN_FILE,
                            MELi, #Callable Object that was instantiated before.
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
                            
    usdTest = SarcasmDataset(TEST_FILE,
                            MELi, #Callable Object that was instantiated before.
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(len(usdTrain))
    print(len(usdTest))
    
    train_data_loader = create_data_loader(usdTrain, BATCH_SIZE)
    test_data_loader = create_data_loader(usdTest, BATCH_SIZE)

    # construct model and assign it to device
    SAi = SelfAttention(9,device).to(device) #Parameter is dim before attention-block
    SAi = SAi.float()
    MelCNNi = MelCNN(SAi).to(device)
    MelCNNi = MelCNNi.float()

    print(MelCNNi)

    # initialise loss funtion + optimiser
    loss_fn = nn.BCELoss()
    #loss_fn = nn.MSELoss()
    optimiser = tr.optim.Adam(MelCNNi.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(MelCNNi, train_data_loader, loss_fn, optimiser, device, EPOCHS, test_data_loader)

    # save model
    tr.save(MelCNNi.state_dict(), "melCNN.pth")
    print("Trained melCNN saved at melCNN.pth")
