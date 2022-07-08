import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import transformers

class MelCNN(nn.Module):
    def __init__(self, selfAttentionBlockInstance, imgChannels=1, dimBeforeAttention=9):
        super(MelCNN, self).__init__()

        self.Conv1 = nn.Conv2d(imgChannels, 3, (5,5), padding=(2,0))
        self.Conv2 = nn.Conv2d(3, dimBeforeAttention, (5,5), padding=(2,0))
        
        #self.query,self.key,self.value = [self._conv(dimBeforeAttention, c) for c in (dimBeforeAttention//8,dimBeforeAttention//8,dimBeforeAttention)]
        #self.gamma = nn.Parameter(torch.tensor([0.]))

        self.attentionBlock = selfAttentionBlockInstance

        self.Conv3 = nn.Conv2d(dimBeforeAttention, 3, (1,1))

        self.Conv4 = nn.Conv2d(3, 27, (5,5), padding=(2,2))
        self.Conv5 = nn.Conv2d(27, 3, (5,5),padding=(2,2))

        self.FC1 = nn.Linear(3*32*1329, 100)
        self.FC2 = nn.Linear(100,1)


    def forward(self, x):
        print("Shape of squeezed tensor: " + str(x.size()))
        x = torch.unsqueeze(x,1) #neded?
        print("Shape of unsqueezed tensor: " + str(x.size()))

        #first conv block
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))

        print('Shape of feature-map before attention-block: ' + str(x.size()))
        plt.figure()
        plt.imshow(x.detach().numpy()[0,0,:,:])
        plt.show()

        x = self.attentionBlock.forward(x)

        print('Shape after attention-block: ' + str(x.size()))
        plt.figure()
        plt.imshow(x.detach().numpy()[0,0,:,:])
        plt.show()

        x = self.Conv3(x) #Pointwise conv makes 3 channels from 9
        print('Shape after attention-block and 1x1conv: ' + str(x.size()))

        #Residual Block
        res = x
        x = F.relu(self.Conv4(x))
        x = F.relu(self.Conv5(x))
        x = x + res
        print('Shape after residual-block: ' + str(x.size()))

        #Flatten
        x = x.view(-1,1,3*24*1329)
        #MLP
        x = F.relu(self.FC1(x))
        x = F.sigmoid(self.FC2(x))


        return x

    def _conv(self,n_in,n_out):
        #return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=nn.NormType.Spectral, act_cls=None, bias=False)
        layer = nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=(1,1), bias=False))
        return layer


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, device):
        super(SelfAttention, self).__init__()
        self.device = device
        self.query,self.key,self.value = [nn.utils.spectral_norm(self._conv(n_channels, c)) for c in (n_channels//9,n_channels//9,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        layer = nn.Conv2d(n_in, n_out, kernel_size=1, bias=False)
        #layer = transformers.modeling_utils.Conv1D(nx=n_in,nf=n_out)
        layer.to(self.device)
        return layer

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        print(x.size())

        #Perform 1x1 convolutions
        f = self.query(x)
        print(f"Size of f after 1x1conv: {f.size()}")
        g = self.key(x)
        h = self.value(x)
        
        #Flatten while keeping the batches. 
        f = f.view(*f.size()[:2],-1)
        g = g.view(*g.size()[:2],-1)
        h = h.view(*h.size()[:2],-1)

        #batched matrix-mult
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)

        #flatten x an add to weighted attention-map
        x = x.view(*size[:2],-1)
        o = self.gamma * torch.bmm(h, beta) + x

        #Restore rectangle shape
        return o.view(*size).contiguous()


if __name__ == "__main__":
    print("convVAE")