from torch import nn
import torch

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=16): #len: 1, 1
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            
            self.make_gen_block(input_dim, hidden_dim * 64, kernel_size=(13, 4)), # 13,4
            self.make_gen_block(hidden_dim * 64, hidden_dim * 32, kernel_size=(13, 4), stride=(1, 1)),# 25, 7
            self.make_gen_block(hidden_dim * 32, hidden_dim * 16, kernel_size=(13, 4), stride=(1, 1)),# 37,10
            self.make_gen_block(hidden_dim * 16, hidden_dim * 8, kernel_size=(13, 4), stride=(2, 2)),# 85,22
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=(13, 4), stride=(1, 1)),# 97,25
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=(13, 4), stride=(1, 1)),# 109,28
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=(13, 4), stride=(1, 1)),# 121,31
            self.make_gen_block(hidden_dim, im_chan, kernel_size=(9, 3), stride=(1, 1), final_layer=True),# 129, 33
            # todo: mejorar arquitectura Red
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''
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
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)
