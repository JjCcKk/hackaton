import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MelCNN(nn.Module):
    def __init__(self, selfAttentionBlockInstance, channel_factor, pool_before_attention, reduce_channels_by, size_hidden_layer, imgChannels=1):
        super(MelCNN, self).__init__()
        
        #Calculate dimensions of model relative to parameters
        self.channel_factor = channel_factor
        self.pool_before_attention = pool_before_attention
        self.reduce_channels_by = reduce_channels_by
        if pool_before_attention:
            height = 5
            self.width_before_mlp = int(332*3*height)
            self.FC1 = nn.Linear(self.width_before_mlp, int(size_hidden_layer*self.width_before_mlp))
            self.FC2 = nn.Linear(int(size_hidden_layer*self.width_before_mlp),1)
        else:
            height = 14
            self.width_before_mlp = int(667*3*height)
            self.FC1 = nn.Linear(self.width_before_mlp, int(size_hidden_layer*self.width_before_mlp))
            self.FC2 = nn.Linear(int(size_hidden_layer*self.width_before_mlp),1)
            
       

        #Initialize Layers
        self.Conv1 = nn.Conv2d(imgChannels, channel_factor, (5,5), padding=(2,2))
        self.Conv2 = nn.Conv2d(channel_factor, channel_factor**2, (5,5), padding=(2,2))

        self.attentionBlock = selfAttentionBlockInstance #Has to be initialized with parameter=min(9,channel_factor**2)

        self.Conv3 = nn.Conv2d(channel_factor**2, int((channel_factor**2)*reduce_channels_by), (1,1))

        self.Conv4 = nn.Conv2d(int((channel_factor**2)*reduce_channels_by), int((channel_factor**2)*reduce_channels_by*9), (5,5), padding=(2,2))
        self.Conv5 = nn.Conv2d(int((channel_factor**2)*reduce_channels_by*9), 3, (5,5),padding=(2,2))

        self.pool = nn.MaxPool2d(5, stride=2)

        


    def forward(self, x):
        #print("Shape of squeezed tensor: " + str(x.size()))
        x = torch.unsqueeze(x,1) #neded?
        #print("Shape of unsqueezed tensor: " + str(x.size()))

        #first conv block
        res1 = x
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = x + res1
        
        if self.pool_before_attention:
            x = self.pool(x) #Turn max pooling of if there is enough memory!!!!!
        
        #print('Shape after max-pooling: ' + str(x.size()))
        #print('Shape of feature-map before attention-block: ' + str(x.size()))

        x = self.attentionBlock.forward(x)

        #print('Shape after attention-block: ' + str(x.size()))
        #plt.figure()
        #plt.imshow(x.cpu().detach().numpy()[0,0,:,:])
        #plt.show()

        x = F.relu(self.Conv3(x)) #Pointwise conv makes 3 channels from 9
        #print('Shape after attention-block and 1x1conv: ' + str(x.size()))

        #Residual Block
        res2 = x
        x = F.relu(self.Conv4(x))
        x = F.relu(self.Conv5(x))
        x = x + res2
        #print('Shape after residual-block: ' + str(x.size()))
        x = self.pool(x)
        #print('Shape after max-pooling: ' + str(x.size()))

        #Flatten
        #x = x.view(-1,1,3*12*int((1329)/2))
        x = x.view(-1,1,self.width_before_mlp)
        #MLP
        x = F.leaky_relu(self.FC1(x))
        x = torch.sigmoid(self.FC2(x))


        return x


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, param, device):
        super(SelfAttention, self).__init__()
        self.device = device
        self.query,self.key,self.value = [nn.utils.spectral_norm(self._conv(n_channels, c)) for c in (n_channels//param,n_channels//param,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        layer = nn.Conv2d(n_in, n_out, kernel_size=1, bias=False)
        #layer = transformers.modeling_utils.Conv1D(nx=n_in,nf=n_out)
        layer.to(self.device)
        return layer

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        #print(x.size())

        #Perform 1x1 convolutions
        f = self.query(x)
        #print(f"Size of f after 1x1conv: {f.size()}")
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