# The blocks used in Generator are defined below
# Only Gated Linear Unit (GLU) and Convolution Block (Convblock) are used in our model.

import torch.nn as nn
# Self Attention module was not used in the generator.
# It was used in many of our trials and generated similar results.
# We did not included in our model to keep our model simple.

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).reshape(-1, self.channels, self.size)


class GLU(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super().__init__()
        self.linear1 = nn.Linear(infeatures, outfeatures)
        self.linear2 = nn.Linear(infeatures, outfeatures)
    def forward(self, x):
        return (self.linear1(x) * self.linear2(x).sigmoid())
    
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, drop=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
            nn.Dropout()
            if drop==True
            else nn.Identity(),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.Mish(out_channels) if use_act else nn.Identity()
            )
    def forward(self, x):
        return (self.conv(x))


# Similar to the Self Attetion block, 
# We did not included Residual blocks in our model to keep it simple.
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)


        