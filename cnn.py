import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

class MelCNN(nn.Module):
    def __init__(self, imgChannels=1, featureDim=16*14*22042, zDim=64):
        super(convVAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 8, (2,5))
        self.encConv2 = nn.Conv2d(8, 16, (2,5))
        #self.encFC1 = nn.Linear(featureDim, zDim)
        #self.encFC2 = nn.Linear(featureDim, zDim)
        #self.encFCtest = nn.Linear(16,128)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        #self.decFCtest = nn.Linear(zDim,128)
        #self.decFC1 = nn.Linear(64, 16)
        self.decConv1 = nn.ConvTranspose2d(16, 8, (2,5))
        self.decConv2 = nn.ConvTranspose2d(8, imgChannels, (2,5))

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        print('Shape after convolutions: ' + str(x.size()))
        x = x.view(-1,1, 16*14*22042)
        print('Shape of view: ' + str(x.size()))
        #x = F.relu(self.encFCtest(x))
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        #x = F.relu(self.decFCtest(z))
        x = F.relu(self.decFC1(z))
        print('Shape after decFC1: ' + str(x.size()))
        x = torch.squeeze(x)
        x = x.view(-1, 16, 14, 22042)
        print('Shape of view decoder: ' + str(x.size()))
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        x = torch.unsqueeze(x,1)
        print("Shape of unsqueezed tensor: " + str(x.size()))


        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


if __name__ == "__main__":
    print("convVAE")