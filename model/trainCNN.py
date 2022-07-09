import torch as tr
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from cnn import MelCNN
from cnn import SelfAttention
from configtrain import TRAIN_FILE, TEST_FILE, TRAIN_ANNOTATION, TEST_ANNOTATION, DESTINATION_FILE
import os
import json

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.00002


def create_data_loader(train_dir, annotation_file, batch_size):
    train_df, label_df = [],[]
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    for i in os.listdir(train_dir):
        vec = np.load(train_dir + i)
        label = data[i.replace(".npy", "")]

        train_df.append(vec)
        label_df.append(label)

    label_df = np.array(label_df)
    train_df = np.array(train_df)

    train_dataloader = DataLoader(train_df, batch_size=batch_size, shuffle=False, drop_last=True)
    label_dataloader = DataLoader(label_df, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, label_dataloader


def train_single_epoch(model, train_data_loader, train_label_data_loader, loss_fn, optimiser, device, test_data_loader, test_label_data_loader):
    for input, lable in zip(train_data_loader, train_label_data_loader):
        input = input.to(device)
        lable = lable.float().to(device)

        if len(lable)==BATCH_SIZE:

            #print("Shape of sample: " + str(input.size()))
            input = input.float()

            # calculate loss
            prediction = model(input)
            #print(prediction.size())
            prediction = tr.squeeze(prediction)
            
            #print(f"Lable: {lable}")
            #print(f"Squeezed Prediction: {prediction}")
            
            loss = loss_fn(prediction,lable)

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    print(f"loss: {loss.item()}")
    #Calculate loss on test set here!
    testLoss = []
    
    #Calculate loss on test-set
    for i,lable in zip(test_data_loader, test_label_data_loader):

        if len(lable)==BATCH_SIZE:

            i = i.to(device)
            lable = lable.bool().int().float().to(device)
            i = i.float()
            # calculate loss
            prediction = model(i)
            prediction = tr.squeeze(prediction)
            loss = loss_fn(prediction,lable)
            testLoss.append(loss.item())
    print(f"test_loss: {np.mean(testLoss)}")


def train(model, train_data_loader, label_train_data_loader, loss_fn, optimiser, device, epochs, test_data_loader, test_label_loader):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, train_data_loader, label_train_data_loader, loss_fn, optimiser, device, test_data_loader, test_label_loader)
        print("---------------------------")
    print("Finished training")
   

if __name__ == "__main__":
    if tr.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    train_data_loader, label_train_data_loader = create_data_loader(TRAIN_FILE, TRAIN_ANNOTATION, BATCH_SIZE)
    test_data_loader, label_test_data_loader = create_data_loader(TEST_FILE, TEST_ANNOTATION, BATCH_SIZE)

    
    # construct model and assign it to device
    ###Select tunable parameters here
    ####################################################
    channel_factor = 2 #Can be [2,4,6]
    pool_before_attention = True #[False, True]
    reduce_channels_by = (1/4) #Can be [1/4, 1/2, 1]
    size_hidden_layer = (1/100) #Can be [(1/10),(1/20), (1/100)]
    ######################################################
    

    #Init instance of self-attention-layer
    SAi = SelfAttention(channel_factor**2, min(9,channel_factor**2),device).to(device) #Parameter is dim before attention-block
    SAi = SAi.float()
    #Init instance of model
    MelCNNi = MelCNN(SAi, channel_factor, pool_before_attention, reduce_channels_by, size_hidden_layer).to(device)
    MelCNNi = MelCNNi.float()

    #Initialise loss funtion + optimiser
    loss_fn = nn.BCELoss()
    #loss_fn = nn.MSELoss()
    optimiser = tr.optim.Adam(MelCNNi.parameters(), lr=LEARNING_RATE)

    # train model
    train(MelCNNi, train_data_loader, label_train_data_loader, loss_fn, optimiser, device, EPOCHS, test_data_loader, label_test_data_loader)
    tr.save(MelCNNi.state_dict(), 'model.pth')
