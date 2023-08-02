import torch.nn as nn
from torch.nn.utils import spectral_norm
from blocks import ConvBlock, GLU, ResidualBlock, SelfAttention
import torch

# The critic network is below:
class Critic(nn.Module):
    def __init__(self, channels):
        super().__init__() 
        self.Mapping_1 = GLU(4096,1024)                                                                         #Nx1x1024
        self.Mapping_2 = GLU(1024,256)                                                                          #Nx1x256
        
        self.down_1 = ConvBlock(channels*1, channels*64, kernel_size=4, stride=2, padding=1)                    #Nx64x128
        self.down_2 = ConvBlock(channels*64, channels*128, kernel_size=4, stride=2, padding=1)                  #Nx128x64
   
        self.last = nn.Conv1d(channels*128, channels*1, kernel_size=64, stride=1, padding=0, padding_mode="reflect") #Nx1x1
        
    def forward(self, x):
        x_1 = self.Mapping_1(x)
        x_2 = self.Mapping_2(x_1)

        x_3 = self.down_1(x_2)
        x_4 = self.down_2(x_3)

        return self.last(x_4)