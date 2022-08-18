from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=16): #len: 1, 1
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            
            self.make_gen_block(input_dim, hidden_dim * 64, kernel_size=(13, 4)), # 13,4
            self.make_gen_block(hidden_dim * 64, hidden_dim * 32, kernel_size=(13, 4), stride=(1, 1)),# 25, 7
            self.make_gen_block(hidden_dim * 32, hidden_dim * 16, kernel_size=(13, 4), stride=(1, 1)),# 37,10
            self.make_gen_block(hidden_dim * 16, hidden_dim * 8, kernel_size=(13, 4), stride=(2, 2)),# 85,22
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=(13, 4), stride=(1, 1)),# 97,25
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=(13, 4), stride=(1, 1)),# 109,28
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=(13, 4), stride=(1, 1)),# 121,31
            self.make_gen_block(hidden_dim, im_chan, kernel_size=(9, 3), stride=(1, 1), final_layer=True),# 129, 33
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                #nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)
