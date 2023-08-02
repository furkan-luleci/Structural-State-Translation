import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch
from blocks import GLU, ConvBlock, ResidualBlock, SelfAttention

# Forming the Generator using the blocks        
class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__() 
        self.Mapping_1 = GLU(4096,1024)                                                                         #Nx1x1024
        self.Mapping_2 = GLU(1024,256)                                                                          #Nx1x256
        
        self.down_1 = ConvBlock(channels*1, channels*64, kernel_size=4, stride=2, padding=1)                    #Nx64x128
        self.down_2 = ConvBlock(channels*64, channels*128, kernel_size=4, stride=2, padding=1)                  #Nx128x64
        
        
    def forward(self, x):
        x_1 = self.Mapping_1(x)
        x_2 = self.Mapping_2(x_1)

        x_3 = self.down_1(x_2)
        x_4 = self.down_2(x_3)
            
        return x_4
 

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__() 
   
        self.up_1 = ConvBlock(channels*128, channels*64, down=False, kernel_size=4, stride=2, padding=1)        #Nx64x128
        self.up_2 = ConvBlock(channels*64, channels*1, down=False, kernel_size=4, stride=2, padding=1)          #Nx1x256

        self.Mapping_3 = GLU(256,1024)                                                                          #Nx1x1024
        self.Mapping_4 = GLU(1024,4096)                                                                         #Nx1x4096
        
	# the last convolution1d(4,2,1) as shown in Fig.8 in the paper is a typo.
	# it should've been convolution1d(7,1,3) as below
        self.last = nn.Conv1d(channels*1, channels*1, kernel_size=7, stride=1, padding=3, padding_mode="reflect")  #Nx1x4096
        
        
    def forward(self, x):

        x_5 = self.up_1(x_4)
        x_6 = self.up_2(x_5+x_3)
        
        x_7 = self.Mapping_3(x_6+x_2)
        x_8 = self.Mapping_4(x_7+x_1)
        
        x_9 = self.last(x_8)
        
        return torch.tanh(x_9)